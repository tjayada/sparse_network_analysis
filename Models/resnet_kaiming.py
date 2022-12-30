# https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
# https://towardsdatascience.com/resnets-for-cifar-10-e63e900524e0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

from Utils.builder import get_builder
from Utils.builder import Builder

from Utils.net_utils import prune


from synflow_args_helper import synflow_parser_args
from gem_miner_args_helper import gem_miner_parser_args

## added from synflow resnet model

class Identity2d(nn.Module):
    def __init__(self, num_features):
        super(Identity2d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features, 1, 1))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode)
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return self._conv_forward(input, W, b)



class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        if self.affine:     
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.affine:
            W = self.weight_mask * self.weight
            b = self.bias_mask * self.bias
        else:
            W = self.weight
            b = self.bias

        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(in_planes, planes, stride=stride)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=1)
        self.bn2 = builder.batchnorm(planes)

        self.shortcut = Identity2d(in_planes)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, planes, kernel_size=1, stride=2, bias=False),
                BatchNorm2d(planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, builder, block, num_blocks, times=1):
        # the times is added, because in smart ratio paper, by default they use twice the channel as standard implementation.
        # so to reproduce the result in their paper, we create this resnet32_double, and set times=2 here
        # by default times is always 1.
        super(ResNet, self).__init__()
        self.in_planes = 16 * times
        self.builder = builder

        
        if gem_miner_parser_args.dataset == "MNIST" or gem_miner_parser_args.dataset == "mnist":
            self.conv1 = builder.conv3x3(1, 16 * times, stride=1, first_layer=True)

        else:
            self.conv1 = builder.conv3x3(3, 16 * times, stride=1, first_layer=True)


        #self.conv1 = builder.conv3x3(3, 16 * times, stride=1, first_layer=True)
        self.bn1 = builder.batchnorm(16 * times)
        self.layer1 = self._make_layer(block, 16 * times, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32 * times, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * times, num_blocks[2], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)

        num_classes = 10
        if gem_miner_parser_args.dataset == "CIFAR100":
            num_classes = 100

        linear_builder = Builder(None, None)
        #self.fc = builder.conv1x1(64 * block.expansion * times, num_classes) # 10 = num_classes for cifar10
        self.fc = linear_builder.linear(64 * block.expansion * times, num_classes) # 10 = num_classes for cifar10

        self.prunable_layer_names, self.prunable_biases = self.get_prunable_param_names()


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def get_prunable_param_names(model):
        prunable_weights = [name + '.weight' for name, module in model.named_modules() if
                isinstance(module, torch.nn.modules.conv.Conv2d) or
                isinstance(module, torch.nn.modules.linear.Linear)]
        if gem_miner_parser_args.bias:
            prunable_biases = [name + '.bias' for name, module in model.named_modules() if
                isinstance(module, torch.nn.modules.conv.Conv2d) or
                isinstance(module, torch.nn.modules.linear.Linear)]
        else:
            prunable_biases = [""]

        return prunable_weights, prunable_biases

    def forward(self, x):
        # update score thresholds for global ep
        if gem_miner_parser_args.algo in ['global_ep', 'global_ep_iter'] or gem_miner_parser_args.bottom_k_on_forward:
            prune(self, update_thresholds_only=True)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        #out = self.fc(out)
        return out #.flatten(1)


def resnet20():
    return ResNet(get_builder(), BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(get_builder(), BasicBlock, [5, 5, 5])


def resnet32_double():
    return ResNet(get_builder(), BasicBlock, [5, 5, 5], 2)


# def resnet44():
#     return ResNet(get_builder(), BasicBlock, [7, 7, 7])


# def resnet56():
#     return ResNet(get_builder(), BasicBlock, [9, 9, 9])


# def resnet110():
#     return ResNet(get_builder(), BasicBlock, [18, 18, 18])


# def resnet1202():
#     return ResNet(get_builder(), BasicBlock, [200, 200, 200])

'''
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)



class BasicBlockNormal(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockNormal, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetNormal(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetNormal, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20Normal():
    return ResNetNormal(BasicBlockNormal, [3, 3, 3])


def resnet32Normal():
    return ResNetNormal(BasicBlockNormal, [5, 5, 5])
'''
