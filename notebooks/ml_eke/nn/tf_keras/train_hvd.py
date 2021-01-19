import nn_models
import loss
import numpy as np  # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

import horovod.keras as hvd
hvd.init()
rank = hvd.rank()


tf.get_logger().setLevel('ERROR')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


X_train = np.load('./data/X_train_sfc.npy')
X_test = np.load('./data/X_test_sfc.npy')

y_train = np.load('./data/y_train_sfc.npy')
y_test = np.load('./data/y_test_sfc.npy')

train_samples = X_train.shape[0]
train_features = X_train.shape[1]
test_samples = X_test.shape[0]

# For fast model research: typically, ~1.0M samples per node is feasible.
train_chunk_size = min(1000000, train_samples//hvd.size())
if rank==0:
    print("Training on {} out of {} training samples"
          .format(train_chunk_size*hvd.size(), train_samples))

X_train = X_train[train_chunk_size*rank:train_chunk_size*(rank+1), :]
y_train = y_train[train_chunk_size*rank:train_chunk_size*(rank+1)]
# X_train = X_train[:train_samples//downsample,:]
# y_train = y_train[:train_samples//downsample]

test_downsample = 100
X_test = X_test[:test_samples//test_downsample, :]
y_test = y_test[:test_samples//test_downsample]


igwloss = loss.InverseGaussianWeightedLoss(y_train)

nominal_lr = 0.001
warmup_epochs = 10

epochs = 1000
batch = 512
lr = nominal_lr * np.sqrt(hvd.size()/8)


def expsched(epoch, lr):
    if epoch < warmup_epochs:
        return nominal_lr/warmup_epochs*(epoch+1.0)*np.sqrt(hvd.size()/8)
    else:
        return lr*0.98


def trisched(epoch, lr):
    lr_max = nominal_lr * np.sqrt(hvd.size()/8)
    if epoch < warmup_epochs:
        return lr_max/warmup_epochs*(epoch+1.0)
    else:
        return lr_max*(1.0-(epoch-warmup_epochs)/(epochs-warmup_epochs+1))


loss_str = 'custom'
loss_f = igwloss.compute_loss if loss_str is 'custom' else loss_str
if loss_f is 'custom' and rank == 0:
  print(str(igwloss))

model = nn_models.build_conv_gen_model(lr=lr, loss=loss_f,
                                       size='XS', train_features=train_features)

filename = "_".join([model.name,
                     str(hvd.size()),
                     str(datetime.today().date()),
                     str(batch),
                     loss_str])+'.h5'
if hvd.rank() == 0:
    model.summary()
    print(f"Training started on {hvd.size()} processes." +
          f"Model will be saved as {filename}")


lr_callback = keras.callbacks.LearningRateScheduler(trisched)
hvd_callback = hvd.callbacks.BroadcastGlobalVariablesCallback(0)

callbacks = [lr_callback, hvd_callback]

steps_per_epoch = int(min(np.ceil(X_train.shape[0]/(batch*hvd.size())), 3200))

if hvd.rank() == 0:
    callbacks.append(
      keras.callbacks.ModelCheckpoint('./checkpoints/{epoch}-'+filename,
                                      save_freq=steps_per_epoch*50))


history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    batch_size=batch,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    verbose=2 if hvd.rank() == 0 else 0,
                    callbacks=callbacks)

if hvd.rank() == 0:
    model.save(filename)
