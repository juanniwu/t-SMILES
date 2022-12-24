import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_, _no_grad_uniform_
#from Models.ModelInit import ModelInit

class ModelInit:
    def Initlize(model, init_type):
        for p in model.parameters():
            if p.dim() > 1:
                if init_type == 'constant':
                    nn.init.constant_(p, 0)
                if init_type == 'ones':
                    nn.init.ones_(p)
                if init_type == 'zeros':
                    nn.init.zeros_(p)
                if init_type == 'eye':
                    nn.init.eye_(p)
                if init_type == 'orthogonal':
                    nn.init.orthogonal_(p)
                if init_type == 'sparse':
                    nn.init.sparse_(p)                
                if init_type == 'xavier_uniform':  #sigmoid、tanh
                    nn.init.xavier_uniform_(p)
                elif init_type == 'xavier_normal': #sigmoid、tanh
                    nn.init.xavier_normal_(p)
                elif init_type == 'kaiming_uniform': #relu and related
                    nn.init.kaiming_uniform_(p)
                elif init_type == 'kaiming_normal': #relu and related
                    nn.init.kaiming_normal_(p)
                elif init_type == 'small_normal_init':
                    ModelInit.xavier_normal_small_init_(p)
                elif init_type == 'small_uniform_init':
                    ModelInit.xavier_uniform_small_init_(p)
        return model

    def xavier_normal_small_init_(tensor, gain=1.):
        # type: (Tensor, float) -> Tensor
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / float(fan_in + 4*fan_out))

        return _no_grad_normal_(tensor, 0., std)


    def xavier_uniform_small_init_(tensor, gain=1.):
        # type: (Tensor, float) -> Tensor
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / float(fan_in + 4*fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

        return _no_grad_uniform_(tensor, -a, a)