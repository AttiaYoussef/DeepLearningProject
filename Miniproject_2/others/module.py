import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import math
import random
torch.set_grad_enabled(False)

class Module(object): # Super class module
    def forward(self, *input):
        pass
    def backward(self, *grad_wrt_output):
        pass
    def params(self):
        return []

class Conv2d(Module):
    
    
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
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
        self.dweights = torch.empty(self.weights.size()).fill_(0)
        self.bias = torch.empty(out_channels) #TODO: initialize
        self.dbias = torch.empty(self.bias.size()).fill_(0)
        
        self.unfolded_input = torch.empty()
        
        ## Keep track of certain values
        self.last_input = None
        self.last_output = None
        self.h_out = 0
        self.w_out = 0
        
    def forward(self, a): 
        self.last_input = a # Input is of shape (N, C, H, W)
        n = a.size(0)
        unfold_a = unfold(a, kernel_size = self.kernel, stride = self.stride, padding = self.padding, dilation = self.dilation)
        self.last_input = unfold_a.clone()
        self.h_out = h_out = math.floor(1 + (a.size(2) + 2 * self.padding[0] - self.dilation[0] * (self.kernel[0] - 1 ) - 1)/self.stride[0])
        self.w_out = w_out = math.floor(1 + (a.size(3) + 2 * self.padding[1] - self.dilation[1] * (self.kernel[1] - 1 ) - 1)/self.stride[1])
        self.last_output = (self.weights.view(self.out_channels,-1) @ unfold_a + self.bias.view(1,-1,1)).view(n, self.out_channels, h_out, w_out)
        return self.last_output # Output is of shape(N, D, H_out, W_out)
    
    def backward(self, grad_wrt_output): #we have dl/ds(l), assume shape (n, D, H_out, W_out). Lecture 3.6 as reference
        correct_shape_grad = grad_wrt_output.view(self.last_output.size(0), self.last_output.size(1), -1)
        grad_wrt_bias = correct_shape_grad.sum(dim=2)
        self.dbias += grad_wrt_bias.sum(dim = 0) #it's cumulative
        
        grad_wrt_weight = correct_shape_grad @ self.last_input.view(self.last_input.size(0), self.last_input.size(2), self.last_input.size(1))
        self.dweights += grad_wrt_weight.sum(dim = 0).view(self.weights.size()) #it's cumulative
        
        grad_wrt_input = (self.weights.view(-1, self.weights.size(0)) @ correct_shape_grad).view(self.last_input.size())
        return grad_wrt_input
    
    def params(self):
        return [(self.weights, self.dweights), (self.bias, self.dbias)]

class TransposedConv2d(Module):
    def __init__(self):
        pass
    
    def forward(self,input):
        pass
    
    def backward(self, *grad_wrt_output):
        pass
    
class MSE(Module):
    def __init__(self,size_average=None, reduce=None, reduction='mean'):
        self.last_input = None
        self.last_target = None
        self.reduction = reduction
    def forward(self, input, target):
        self.last_input = input
        self.last_target = target
        error = ((input - target)**2).sum()
        if self.reduction == 'mean':
            error = error/input.size(0)
        return error
    
    def backward(self, grad_wrt_output): 
        grad_wrt_input = 2 * (self.last_input - self.last_target) #simple derivative
        if self.reduction == 'mean':
            grad_wrt_input /= self.last_input.size(0)
        return grad_wrt_input
    
    def params(self):
        return []
    
class SGD(object): #I think they want it as a module, go figure why
    
    def __init__(self, params, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False):
        self.model_params = model_params
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        
    def forward(self, *inputs):
        pass
    
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
        for module in self.modules[::-1] : #Go from last layer to the first
            grad_wrt_output = module.backward(grad_wrt_output)
    
    def params(self):
        parameters = []
        for module in self.modules:
            parameters += module.parameters()
        return parameters
    
class ReLU(Module):
    def __init__(self, inplace=False):
        self.last_input = None
        self.last_output = None
        self.inplace = inplace
        
    def forward(self, x):
        self.last_input = x
        result = x.clone()
        result[result <= 0] = 0
        self.last_output = result
        return result
    
    def backward(self, grad_wrt_output):
        mask = torch.empty(self.last_input.size()).fill(0)
        mask[self.last_output > 0] = 1
        grad_wrt_input = mask * grad_wrt_output # slide 10 of lecture 3.6
        return grad_wrt_input
    
    def params(self):
        return []
    
class Sigmoid(Module):
    
    self.last_input = None
    self.last_output = None
    def __init__(self):
        pass
    
    def __calculate_sigmoid__(self, x):
        return 1/(1 + (-x).exp())
    def forward(self, x):
        self.last_input = x
        self.last_output = __calculate_sigmoid__(x)
        return self.last_output
    
    def backward(self, grad_wrt_output):
        grad_wrt_input = grad_wrt_output * (self.last_output * (1 - self.last_output)) # slide 10 of lecture 3.6
        return grad_wrt_input
    
    def params(self):
        return []