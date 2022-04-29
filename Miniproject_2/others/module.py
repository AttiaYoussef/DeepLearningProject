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
    
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 0, dilation = 1):
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
        
        self.weights = torch.empty(out_channels,in_channels,self.kernel[0], self.kernel[1]) #TODO: initialize uniformly
        self.bias = torch.empty(out_channels) #TODO: initialize uniformly
            
    def forward(self, a):
        n = a.size(0)
        unfold_a = unfold(a, kernel_size = self.kernel, stride = self.stride, padding = self.padding, dilation = self.dilation)
        h_out = math.floor(1 + (a.size(2) + 2 * self.padding[0] - self.dilation[0] * (self.kernel[0] - 1 ) - 1)/self.stride[0])
        w_out = math.floor(1 + (a.size(3) + 2 * self.padding[1] - self.dilation[1] * (self.kernel[1] - 1 ) - 1)/self.stride[1])
        return (self.weights.view(self.out_channels,-1) @ unfold_a + self.bias.view(1,-1,1)).view(n, self.out_channels, h_out, w_out)
    
    def backward(self, *gradwrtoutput):
        pass
    
    def params(self):
        return [self.weights, self.bias]
    
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
        parameters = []
        for module in self.modules:
            parameters += module.parameters()
        return parameters()
    
class ReLU(Module):    
    def forward(self, x):
        x[x <= 0] = 0
        self.x = x
        return x
    
    def backward(self, *gradwrtoutput):
        pass
    
    def params(self):
        return []
    
class Sigmoid(Module):
    def forward(self, x):
        return 1/(1 + torch.exp(-x))
    
    def backward(self, *gradwrtoutput):
        pass
    
    def params(self):
        return []