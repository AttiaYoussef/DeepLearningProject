import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import math
import random
torch.set_grad_enabled(False)

class Module(object):
    def forward(self, *input):
        pass
    def backward(self, *gradwrtoutput):
        pass
    def params(self):
        return []

class Conv2d(Module):
    
    def __init__(self, in_channels, out_channels, kernel, stride,padding,dilation):
        super().__init__()
        
        def __parameter_int_or_tuple__(parameter):
            if type(parameter) is int:
                returned = (parameter,parameter)
            else:
                returned = parameter
            return returned
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = __parameter_int_or_tuple__(kernel)
        self.stride = __parameter_int_or_tuple__(stride)
        self.padding = __parameter_int_or_tuple__(padding)
        self.dilation = __parameter_int_or_tuple__(dilation)
        
        self.weights = torch.empty(out_channels,self.kernel[0], self.kernel[1])
        self.bias = torch.empty(out_channels)
            
    def forward(self, x):
        pass
    
    def backward(self, *gradwrtoutput):
        pass
    
    def params(self):
        return []
    
class MSELoss(Module):
    def forward(self, input, target):
        return ((input - target)**2).mean()
    
    def backward(self, *gradwrtoutput):
        pass
    
    def params(self):
        return []
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
    
    def forward(self, *x):
        for module in self.modules:
            x = module.forward(x)
            
    def backward(self, *gradwrtoutput):
        pass
    
    def params(self):
        return []
class ReLU(Module):
    def forward(self, x):
        x[x <= 0] = 0
        return x
    
    def backward(self, *gradwrtoutput):
        pass
    
    def params(self):
        return []
    
class Sigmoid(Module):
    def forward(self, x):
        pass
    
    def backward(self, *gradwrtoutput):
        pass
    
    def params(self):
        return []