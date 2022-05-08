import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import math
import random
torch.set_grad_enabled(False)

class Module(object):
    def forward(self, *input):
        pass
    def backward(self, *grad_wrt_output):
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
        
        self.weights = torch.empty(out_channels,in_channels,self.kernel[0], self.kernel[1]) #TODO: initialize
        self.dweights = torch.empty(self.weights.size())
        self.bias = torch.empty(out_channels) #TODO: initialize
        self.dbias = torch.empty(self.bias.size())
        
        self.unfolded_input = torch.empty()
        
        ## Keep track of certain values
        self.last_input = 0    
        self.last_output = 0
        self.h_out = 0
        self.w_out = 0
        
    def forward(self, a):
        self.last_input = a # Input is of shape (N, C, H, W)
        n = a.size(0)
        unfold_a = unfold(a, kernel_size = self.kernel, stride = self.stride, padding = self.padding, dilation = self.dilation)
        self.last_input = torch.copy(unfold_a)
        self.h_out = h_out = math.floor(1 + (a.size(2) + 2 * self.padding[0] - self.dilation[0] * (self.kernel[0] - 1 ) - 1)/self.stride[0])
        self.w_out = w_out = math.floor(1 + (a.size(3) + 2 * self.padding[1] - self.dilation[1] * (self.kernel[1] - 1 ) - 1)/self.stride[1])
        self.last_output = (self.weights.view(self.out_channels,-1) @ unfold_a + self.bias.view(1,-1,1)).view(n, self.out_channels, h_out, w_out)
        return self.last_output # Output is of shape(N, D, H_out, W_out)
    
    def backward(self, grad_wrt_output): #we have dl/ds(l), assume shape (n, D, H_out, W_out)
        correct_shape_grad = grad_wrt_output.view(self.last_output.size(0), self.last_output.size(1), -1)
        self.dbias = torch.sum(correct_shape_grad, axis=2)
        self.dbias = torch.sum(self.dbias, axis=0)
        
        self.dweights = correct_shape_grad @ self.last_input.view(self.last_input.size(0), self.last_input.size(2), self.last_input.size(1))
        self.dweights = torch.sum(self.dweights, axis=0)
        
        pass
    
    def params(self):
        return [(self.weights, self.dweights), (self.bias, self.dbias)]
    
class MSELoss(Module):
    def forward(self, input, target):
        error = 0
        for n in range(input.size(0)):
            for c in range(input.size(1)):
                for row in range(input.size(2)):
                    row_x = input[n,c,row]
                    row_y = target[n,c,row]
                    error += sum((row_x-row_y)**2)
        return error/input.size(0)
    
    def backward(self, *grad_wrt_output):
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
            
    def backward(self, *grad_wrt_output):
        pass
    
    def params(self):
        parameters = []
        for module in self.modules:
            parameters += module.parameters()
        return parameters()
    
class ReLU(Module):
    def __init__(self):
        self.output = 0
    def forward(self, x):
        result = x.clone()
        result[result <= 0] = 0
        self.output = result
        return result
    def backward(self, *grad_wrt_output):
        pass
    
    def params(self):
        return []
    
class Sigmoid(Module):
    def forward(self, x):
        return 1/(1 + torch.exp(-x))
    
    def backward(self, *grad_wrt_output):
        pass
    
    def params(self):
        return []