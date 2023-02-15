import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)        
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return F.linear(input, W, b)

# structure hard-coded
class rnd_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, sparse = 1):
        super(rnd_Linear, self).__init__()        
        first_layer = False
        hidden_layer = False
        self.weight = Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty((out_features)))
        else:
            self.register_parameter('bias', None)
        
        rnd_mask = torch.zeros(self.weight.shape)
        
        # for the amount of parameters that should remain do ..
        # ps. not the best solution, since hard to know intuitively and multiple weights are placed down, thus not exact anyways
        for _ in range(sparse):
            same = True
            # in case a weight is placed in a position already occupied, try again, until position is not same
            while same:
                # differentiate the different layers with the in and out features they are defined with
                if in_features > 700:
                    first_layer = True
                    # only use the first 6 computational units, and randomly assign weigths
                    rnd_x, rnd_y = torch.randint(6,(1,)), torch.randint(self.weight.shape[1],(1,))
                
                elif out_features == 10:
                    # use all computational units and randomly assign weigths
                    rnd_x, rnd_y = torch.randint(self.weight.shape[0],(1,)), torch.randint(self.weight.shape[1],(1,))
                
                else:
                    hidden_layer = True
                    # only use the first 30 computational units, and randomly assign weigths
                    rnd_x, rnd_y = torch.randint(30,(1,)), torch.randint(self.weight.shape[1],(1,))

                try:
                    # check whether weight position is already occupied
                    if rnd_mask[rnd_x, rnd_y] == 0:
                        same = False
                        # if not place weight, also in neighbouring unit
                        rnd_mask[rnd_x, rnd_y] = 1
                        rnd_mask[rnd_x + 1, rnd_y] = 1
                        
                        # check whether weight position of neighbour is already occupied
                        if rnd_mask[rnd_x, rnd_y + 1] == 0:
                            rnd_mask[rnd_x, rnd_y + 1] = 1
                            
                            if rnd_mask[rnd_x, rnd_y + 2] == 0:
                                rnd_mask[rnd_x, rnd_y + 2] = 1
            
                except:
                    pass

        self.register_buffer('weight_mask', rnd_mask)
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

        self.reset_parameters(first_layer, hidden_layer)



    def reset_parameters(self, first_layer, hidden_layer) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        
        # has not been tested yet
        # idea is to use mostly positive weights, as indicated by Gem-Miner results
        # thus after same initialization as always, find negative weights and make positive
        # but not all, only big fraction, and also depnendent on layer, since first layer mostly
        # negative and hidden mostly positive and last equal
        """
        if first_layer:
            x,y = np.argwhere(self.weight>0)

            for i in range(len(x)):
                if np.random.randint(3) > 1:
                    self.weight[x[i]][y[i]] = -1 * self.weight[x[i]][y[i]]

        elif hidden_layer:
            x,y = np.argwhere(self.weight<0)

            for i in range(len(x)):
                if np.random.randint(2) == 1:
                    self.weight[x[i]][y[i]] = abs(self.weight[x[i]][y[i]])

        """

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return F.linear(input, W, b)



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


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm1d, self).__init__(
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


class Identity1d(nn.Module):
    def __init__(self, num_features):
        super(Identity1d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W


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



