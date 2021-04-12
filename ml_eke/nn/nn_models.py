import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Union, List, Optional, Tuple
from torch.nn.functional import silu

from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

# activation_def = 'relu'
# kernel_initializer_def = 'glorot_normal'
# L2 = 2e-3
# kernel_regularizer_def = regularizers.l2(L2)
# bias_regularizer_def = regularizers.l2(L2)
# activity_regularizer_def = regularizers.l2(L2)
X_train_shape_1 = 10


def next_pow2(x):
    return int(np.ceil(np.log2(np.abs(x))))


class TransBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        filters: int = 256,
        size: Tuple[int, int] = (3, 3)
    ) -> None:

        super(TransBlock, self).__init__()

        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=filters,
                                            kernel_size=size,
                                            stride=1,
                                            padding=0,
                                            output_padding=0)

        self.bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.convTrans(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class EKETriNet(nn.Module):

    def __init__(
            self,
            in_features: int,
            depth: int = 10,
            out_features: int = 1
            ):

        super(EKETriNet, self).__init__()
        self.name = f'TriNet_{in_features}_{depth}'

        if in_features==2**next_pow2(in_features):
            self.ispow2 = 1
        else:
            self.ispow2 = 0

        self.depth = depth
        self.sizes = [2**next_pow2(in_features)*2**(i) for i in range(depth+1)]
        if not self.ispow2:
            self.sizes[0] = in_features

        self.trilayers = (nn.ModuleList([nn.Linear(self.sizes[i], self.sizes[i+1]) for i in range(depth)]))
        self.trilayers.extend(nn.ModuleList([nn.Linear(self.sizes[i+1], self.sizes[i]) for i in reversed(range(depth))]))

        self.fc = nn.Linear(self.sizes[0], out_features)

        
    def forward(self, x: Tensor) -> Tensor:
        xs = [x]
        for i in range(1, self.depth+1):
            x = silu(self.trilayers[i-1](x))
            xs = xs + [x]
        
        for i in range(self.depth):
            x = silu(self.trilayers[i+self.depth](x) + xs[self.depth-i-1])

        x = self.fc(x)
        
        return x

class EKEWideTriNet(nn.Module):

    def __init__(
            self,
            in_features,
            depth: int = 4,
            width: int = 4,
            trinet_out_features: int = 1,
            out_features: int = 1,
            ):

        super(EKEWideTriNet, self).__init__()
        self.name = f'WideTriNet_{in_features}_{depth}x{width}'
        self.depth = depth
        self.width = width

        self.trinets = nn.ModuleList([EKETriNet(in_features, depth, trinet_out_features) for i in range(width)])

        self.fc = nn.Linear(in_features+width*trinet_out_features, out_features)


    def forward(self, x: Tensor) -> Tensor:
        xs = x
        for i in range(self.width):
            xs = torch.cat((xs, self.trinets[i](x)), 1)

        x = self.fc(silu(xs))
        return x

class EKEResnet(nn.Module):
    ''' Just a very small resnet-like network, implemented starting from
        PyTorch's implementation of ResNet
    '''
    def __init__(
        self,
        train_features: int,
        num_classes: int = 1,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super(EKEResnet, self).__init__()
        self.name = f'ResNet_{train_features}'
        self.norm_layer = nn.BatchNorm2d

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}"
                             .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.train_features = train_features

        self.transBlock1 = TransBlock(train_features, X_train_shape_1*2)
        self.transBlock2 = TransBlock(X_train_shape_1*2,
                                      2*2**next_pow2(11))
        num_in_filters = 2**next_pow2(11)*2
        self.transBlock3 = TransBlock(num_in_filters,
                                      2*2**next_pow2(11))

        block = BasicBlock

        self.layer1 = self._make_layer(Bottleneck, 16, 1)
        self.maxpool = nn.AdaptiveMaxPool2d((4, 4))
        self.layer2 = self._make_layer(Bottleneck, 32, 1)
        self.layer3 = self._make_layer(Bottleneck, 32, 1)

        self.fc1 = nn.Linear(256*4*2, 32)
        self.fc2 = nn.Linear(32, 8)

        self.fc3 = nn.Linear(8, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according
        # to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type] #noqa
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type] #noqa

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        x = torch.reshape(x, (-1,self.train_features, 1, 1))
        x = self.transBlock1(x)
        x = self.transBlock2(x)
        x = self.transBlock3(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class EKEResnetSmall(nn.Module):
    ''' Just a very small resnet-like network, implemented starting from
        PyTorch's implementation of ResNet
    '''
    def __init__(
        self,
        train_features: int,
        num_classes: int = 1,
        zero_init_residual: bool = False,
        groups: int = 4,
        width_per_group: int = 4,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super(EKEResnetSmall, self).__init__()
        self.name = f'ResNetSmall_{train_features}'
        self.norm_layer = nn.BatchNorm2d

        self.inplanes = 8
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}"
                             .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.train_features = train_features

        out_first_layer = 8

        self.transBlock1 = TransBlock(self.train_features, out_first_layer, size=(2,2))
        self.transBlock2 = TransBlock(out_first_layer, out_first_layer, size=(2,2))
        self.transBlock3 = TransBlock(out_first_layer, out_first_layer, size=(2,2))

        #block = BasicBlock

        self.layer1 = self._make_layer(Bottleneck, 16, 2)
        self.maxpool1 = nn.AdaptiveMaxPool2d((2, 2))
        self.layer2 = self._make_layer(Bottleneck, 16, 2)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according
        # to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type] #noqa
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type] #noqa

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        x = torch.reshape(x, (-1,self.train_features, 1, 1))
        x = self.transBlock1(x)
        x = self.transBlock2(x)
        x = self.transBlock3(x)

        x = self.layer1(x)
        x = self.maxpool1(x)
        x = self.layer2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class EKECNN(nn.Module):

    def __init__(
            self,
            train_features: int,
            num_classes: int = 1,
            groups:int = 4,
            width_per_group = 2
            ) -> None:
        super(EKECNN, self).__init__()
        self.name = f'CNN_{train_features}_{groups}x{width_per_group}'

        self.train_features = train_features
        self.num_classes = num_classes
        self.post_adapt = groups*width_per_group
        expansion = 2
        self.adapter = nn.Conv2d(in_channels = train_features,
                                 out_channels = self.post_adapt, 
                                 kernel_size = 1,
                                 groups = 1)

        self.conv1 = nn.Conv2d(in_channels = self.post_adapt,
                               out_channels = self.post_adapt,
                               kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(self.post_adapt)
        self.conv2 = nn.Conv2d(self.post_adapt, self.post_adapt*expansion, 1, groups)
        self.bn2 = nn.BatchNorm2d(self.post_adapt*expansion)
        self.conv3 = nn.Conv2d(self.post_adapt*expansion, self.post_adapt, 1, groups)
        self.bn3 = nn.BatchNorm2d(self.post_adapt)
        self.relu = nn.ReLU(inplace=False)
        #self.final_adapter = nn.Conv2d(post_adapt, num_classes, 1)
        self.fc1 = nn.Linear(self.post_adapt, 4)
        self.fc2 = nn.Linear(4, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.bn3.weight, 0)  # type: ignore[arg-type] #noqa
        

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        x = torch.reshape(x, (-1, self.train_features, 1, 1))
        adapted = self.adapter(x)

        out = self.conv1(adapted)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        
        out += adapted
        out = self.relu(out)
        #out = self.final_adapter(out)
        
        out = torch.reshape(out, (-1, self.post_adapt))
        out = self.fc1(out)
        out = self.fc2(out)

        out = torch.flatten(out)
    
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class EKEResnetExtraSmall(nn.Module):
    ''' Just a very small resnet-like network, implemented starting from
        PyTorch's implementation of ResNet
    '''
    def __init__(
        self,
        train_features: int,
        num_classes: int = 1,
        zero_init_residual: bool = False,
        groups: int = 4,
        width_per_group: int = 8,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super(EKEResnetExtraSmall, self).__init__()
        self.name = f'ResNetExtraSmall_{train_features}'
        self.norm_layer = nn.BatchNorm2d

        self.inplanes = 4
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}"
                             .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.train_features = train_features


        self.transBlock1 = TransBlock(self.train_features, self.inplanes, size=(2,2))
        self.transBlock2 = TransBlock(self.inplanes, self.inplanes, size=(2,2))
        self.transBlock3 = TransBlock(self.inplanes, self.inplanes, size=(2,2))


        self.layer1 = self._make_layer(Bottleneck, 8, 2)
        self.maxpool1 = nn.AdaptiveMaxPool2d((2, 2))

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according
        # to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type] #noqa
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type] #noqa

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        x = torch.reshape(x, (-1,self.train_features, 1, 1))
        x = self.transBlock1(x)
        x = self.transBlock2(x)
        x = self.transBlock3(x)

        x = self.layer1(x)
        x = self.maxpool1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
