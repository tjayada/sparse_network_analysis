import torch
import torch.nn as nn
import numpy as np
from Layers import layers
from torch.nn import functional as F

from Utils.net_utils import prune
from gem_miner_args_helper import gem_miner_parser_args

from Utils.builder import Builder


def fc(input_shape, num_classes, dense_classifier=False, pretrained=False, L=6, N=100, nonlinearity=nn.ReLU()):
  size = np.prod(input_shape)
  
  # Linear feature extractor
  modules = [nn.Flatten()]
  modules.append(layers.Linear(size, N))
  modules.append(nonlinearity)
  for i in range(L-2):
      modules.append(layers.Linear(N,N))
      modules.append(nonlinearity)

  # Linear classifier
  if dense_classifier:
      modules.append(nn.Linear(N, num_classes, bias=False))
  else:
      modules.append(layers.Linear(N, num_classes, bias=False))
  model = nn.Sequential(*modules)

  # Pretrained model
  if pretrained:
      print("WARNING: this model does not have pretrained weights.")
  
  return model



class FC(nn.Module):
    def __init__(self, input_shape, num_classes, L=6, N=100, nonlinearity=nn.ReLU()):
        super(FC, self).__init__()
        
        model_builder = Builder(None, None)
        size = np.prod(input_shape)

        # Linear feature extractor
        modules = [nn.Flatten()]
        modules.append(model_builder.linear(size, N))
        modules.append(nonlinearity)
        for i in range(L-2):
            modules.append(model_builder.linear(N,N))
            modules.append(nonlinearity)
        
        modules.append(model_builder.linear(N, num_classes))
        self.model = nn.Sequential(*modules)



    def forward(self, x):
        #x = x.view(x.size()[0], -1)
        x = self.model(x)
        output = F.log_softmax(x, dim=1)
        #output = x

        return output




