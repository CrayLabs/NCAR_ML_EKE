import argparse
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.data import TensorDataset
import horovod.torch as hvd
import numpy as np
from scipy.stats import norm
from torchsummary import summary
import time

from nn_models import EKEResnet, EKETriNet, EKEWideTriNet, EKEResnetSmall

# Training settings
parser = argparse.ArgumentParser(description='PyTorch EKE Training with HVD')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--model', type=str, default='resnet', metavar='N',
                    help='model type (default: resnet)', choices=['resnet', 'resnet_small', 'trinet', 'widetrinet'])
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--weighted-sampling', action='store_true', default=False,
                    help='use weighted sampling for training data')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')



def train(epoch):
    model.train()
    if not args.weighted_sampling:
        # Horovod: set epoch to sampler for shuffling.
        train_sampler.set_epoch(epoch)
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss_fn = nn.MSELoss()
        loss = loss_fn(output.squeeze(), target.squeeze())
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_idx % args.log_interval == 0 and rank==0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('{} - Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} - elapsed time: {:.2f}'.format(
                model.name, epoch, (batch_idx+1) * len(data), len(train_sampler),
                100. * (batch_idx+1) / len(train_loader), loss.item(), time.time()-start))
    if rank==0:
        print(f'Epoch time: {time.time()-start}')


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test(epoch):
    model.eval()
    test_loss = 0.
    loss_fn = nn.MSELoss(reduction = 'sum')
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += loss_fn(output.squeeze(), target.squeeze()).item()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)


    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\n{} - Train epoch: {} test set: Average loss: {:.4f}\n'.format(
            model.name, epoch, test_loss))


def compute_weights(samples):
    mu, std = norm.fit(samples)
    std_vec = std * torch.ones_like(samples)
    mu_vec  = mu  * torch.ones_like(samples)
    pi_vec  = np.pi    * torch.ones_like(samples)

    weights = ((std_vec*torch.sqrt(2.0*pi_vec)) *
               torch.exp(0.5*torch.square((samples-mu_vec)/std_vec)))

    print(torch.min(weights), torch.max(weights))
    weights = torch.clamp(weights, 0, 1000)

    return weights

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # Horovod: initialize library.
    hvd.init()
    rank = hvd.rank()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)


    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    X_train = np.load('../data/X_train_prep_cf23.npy')
    X_test = np.load('../data/X_test_prep_cf23.npy')

    y_train = np.load('../data/y_train_prep_cf23.npy')
    y_test = np.load('../data/y_test_prep_cf23.npy')

    train_samples = X_train.shape[0]
    train_features = X_train.shape[1]
    test_samples = X_test.shape[0]

    # For fast model research: typically, ~1.0M samples per node is feasible.
    max_samples = 1000000
    if args.weighted_sampling:
        max_samples *= 10  # allow to select 1/10 of samples according to weights
    train_size = min(max_samples*hvd.size(), train_samples)//hvd.size()*hvd.size()

    X_train = torch.tensor(X_train[:train_size, :])
    y_train = torch.tensor(y_train[:train_size])

    test_downsample = 100
    test_samples = (test_samples//test_downsample)
    X_test = torch.tensor(X_test[:test_samples, :])
    y_test = torch.tensor(y_test[:test_samples])

    if args.weighted_sampling:
        train_chunk_size = train_size // hvd.size()
        X_train = X_train[rank*train_chunk_size:(rank+1)*train_chunk_size,:]
        y_train = y_train[rank*train_chunk_size:(rank+1)*train_chunk_size]
        weights = compute_weights(y_train)
        train_dataset = TensorDataset(X_train, y_train)
        train_sampler = torch.utils.data.WeightedRandomSampler(weights, train_chunk_size//10,
                replacement=False)
        if rank==0:
          print("Training on {} out of {} training samples"
                .format(len(y_train)*hvd.size(), train_samples))
    else:
        train_dataset = TensorDataset(X_train, y_train)
        # Horovod: use DistributedSampler to partition the training data.
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    
        if rank==0:
          print("Training on {} out of {} training samples"
                .format(train_size, train_samples))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    test_dataset = TensorDataset(X_test, y_test)
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)

    weight_decay = 2e-4
    if args.model.lower() == 'trinet':
        model = EKETriNet(train_features, 8)
    elif args.model.lower() == 'resnet':
        model = EKEResnet(train_features)
        weight_decay = 2e-3
    elif args.model.lower() == 'widetrinet':
        model = EKEWideTriNet(train_features, depth=4, width=8) # best 3x4
    elif args.model.lower() == 'resnet_small':
        model = EKEResnetSmall(train_features)

    if rank==0:
        print(model.name)

    if args.weighted_sampling:
        args.lr /= 10.0

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        if rank==0:
            print("CUDA found.")
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
                          momentum=args.momentum, weight_decay=weight_decay)
    

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none


    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average,
                                         gradient_predivide_factor=args.gradient_predivide_factor)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr*lr_scaler,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs, pct_start=5./args.epochs)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if rank==0 and epoch%10 == 0 and epoch>0:
            loss_str = 'custom' if args.weighted_sampling else 'mse'
            torch.save(model, f'{model.name}-{epoch}_{loss_str}_prep_cf23.pkl')
        test(epoch)

