import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import math
import random
#torch.set_grad_enabled(False)

def __parameter_int_or_tuple__(parameter):
    if type(parameter) is int:
        returned = (parameter,parameter)
    else:
        returned = parameter
    return returned

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
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = __parameter_int_or_tuple__(kernel)
        self.stride = __parameter_int_or_tuple__(stride)
        self.padding = __parameter_int_or_tuple__(padding)
        self.dilation = __parameter_int_or_tuple__(dilation)
        
        bound = math.sqrt(groups/(in_channels * self.kernel[0] * self.kernel[1]))
        self.weights = torch.empty(out_channels, in_channels, self.kernel[0], self.kernel[1]).uniform_(-bound, bound)
        self.dweights = torch.empty(self.weights.size()).fill_(0)
        self.bias = torch.empty(out_channels).uniform_(-bound, bound)
        self.dbias = torch.empty(self.bias.size()).fill_(0)
        
        ## Keep track of certain values
        self.last_input_size = None
        self.last_input = None
        self.last_output = None
        self.h_out = 0
        self.w_out = 0
        
    def forward(self, a): 
        self.last_input_size = a.size() # Input is of shape (N, C, H, W)
        n = a.size(0)
        unfold_a = unfold(a, kernel_size = self.kernel, stride = self.stride, padding = self.padding, dilation = self.dilation)
        
        self.last_input = unfold_a.clone()
        self.h_out = h_out = math.floor(1 + (a.size(2) + 2 * self.padding[0] - self.dilation[0] * (self.kernel[0] - 1 ) - 1)/self.stride[0])
        self.w_out = w_out = math.floor(1 + (a.size(3) + 2 * self.padding[1] - self.dilation[1] * (self.kernel[1] - 1 ) - 1)/self.stride[1])
        self.last_output = (self.weights.view(self.out_channels,-1) @ unfold_a + self.bias.view(1,-1,1)).view(n, self.out_channels, h_out, w_out)
        return self.last_output # Output is of shape(N, D, H_out, W_out)
    
    def backward(self, grad_wrt_output): #we have dl/ds(l), assume shape (n, D, H_out, W_out). Lecture 3.6 as reference
        correct_shape_grad = grad_wrt_output.view(self.last_output.size(0), self.last_output.size(1), -1)
        grad_wrt_bias = grad_wrt_output.sum(dim=[0,2,3])
        self.dbias += grad_wrt_bias #it's cumulative 
        
        
        def spicy_reshape(x):
            return x.transpose(0,1).transpose(1,2).reshape(x.shape[1],x.shape[0]*x.shape[2])
        
        
        dout_reshaped = grad_wrt_output.transpose(0,1).transpose(1,2).transpose(2,3).reshape(self.out_channels, -1)
        dW = dout_reshaped @ spicy_reshape(self.last_input).T
        self.dweights += dW.reshape(self.weights.shape)
        
        #grad_wrt_weight = correct_shape_grad @ self.last_input.view(self.last_input.size(0), self.last_input.size(2), self.last_input.size(1))
        #self.dweights += grad_wrt_weight.sum(dim = 0).view(self.weights.size()) #it's cumulative
        
        grad_wrt_input = (self.weights.view(self.weights.size(0), -1).T @ correct_shape_grad).view(self.last_input.size())
        grad_wrt_input = fold(grad_wrt_input, output_size = self.last_input_size[2:], kernel_size = self.kernel,dilation=self.dilation, padding=self.padding, stride=self.stride)
        return grad_wrt_input
    
    def params(self):
        return [(self.weights, self.dweights), (self.bias, self.dbias)]

class NearestUpsampling(Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        if size is not None :
            self.scale = __parameter_int_or_tuple__(size)
        elif scale_factor is not None :
            self.scale = __parameter_int_or_tuple__(scale_factor)
    
    def forward(self,input):
        return input.repeat_interleave(self.scale[0], dim=2).repeat_interleave(self.scale[1], dim=3)
    
    def backward(self, grad_wrt_output):
        acc=torch.zeros(1,grad_wrt_output.size(0),grad_wrt_output.size(1),int(grad_wrt_output.size(2)/self.scale[0]),int(grad_wrt_output.size(3)/self.scale[1]))
        
        for i in range(self.scale[0]):
            for j in range(self.scale[1]):
                acc=torch.cat((acc,grad_wrt_output[:,:,i::self.scale[0],j::self.scale[1]].\
                               view(1,grad_wrt_output.size(0),grad_wrt_output.size(1),int(grad_wrt_output.size(2)/self.scale[0]),int(grad_wrt_output.size(3)/self.scale[1]))),dim=0)
        
        
        return acc.sum(dim=0)
    
class MSE(Module):
    def __init__(self,size_average=None, reduce=None, reduction='mean'):
        self.last_input = None
        self.last_target = None
        self.reduction = reduction
    def forward(self, input, target):
        self.last_input = input
        self.last_target = target
        error = ((input - target)**2)
        if self.reduction == 'mean':
            error = error.mean()
        else:
            error = error.sum()
        return error
    
    def backward(self): 
        grad_wrt_input = 2 * (self.last_input - self.last_target) #simple derivative
        if self.reduction == 'mean':
            grad_wrt_input = grad_wrt_input/(self.last_input.size(0)*self.last_input.size(1)*self.last_input.size(2)*self.last_input.size(3))
        return grad_wrt_input
    
    def params(self):
        return []
    
class SGD(object):
    
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False):
        self.model_params = params
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        
        # parameters needed in the case of momentum
        self.b_ts = []
        self.previous_grads = []
        for (_, _) in self.model_params:
            self.b_ts.append(None)
            self.previous_grads.append(0)
        
        
    def step(self): # see SGD on pytorch
        for index, (module_param, module_param_grad) in enumerate(self.model_params):
            if self.weight_decay != 0:
                module_param_grad += self.weight_decay * module_param
            if self.momentum != 0:
                if self.b_ts[index] is None:
                    self.b_ts[index] = module_param_grad
                else:
                    self.b_ts[index] = self.momentum * self.b_ts[index] + (1 - self.dampening) * module_param_grad
                
                if self.nesterov:
                    module_param_grad = self.previous_grads[index] + self.momentum * self.b_ts[index]
                else:
                    module_param_grad = self.b_ts[index]
                
                self.previous_grads[index] = module_param_grad
                
            if self.maximize:
                module_param += self.lr * module_param_grad
            else:
                module_param -= self.lr * module_param_grad
    
    def zero_grad(self):
        for i, (_, module_param_grad) in enumerate(self.model_params):
            self.b_ts[i] = None
            self.previous_grads[i] = 0
            module_param_grad.zero_()
        
    
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
    
    def forward(self, x):
        y = torch.clone(x)
        for module in self.modules:
            y = module.forward(y)
        return y
            
    def backward(self, grad_wrt_output):
        for module in self.modules[::-1] : #Go from last layer to the first
            
            grad_wrt_output = module.backward(grad_wrt_output)
    
    def params(self):
        parameters = []
        for module in self.modules:
            parameters += module.params()
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
        mask = torch.empty(self.last_input.size()).fill_(0)
        mask[self.last_output > 0] = 1
        grad_wrt_input = mask * grad_wrt_output # slide 10 of lecture 3.6
        return grad_wrt_input
    
    def params(self):
        return []
    
class Sigmoid(Module):
    def __init__(self):
        self.last_input = None
        self.last_output = None
    
    def __calculate_sigmoid__(self, x):
        return 1/(1 + (-x).exp())
    
    def forward(self, x):
        self.last_input = x
        self.last_output = 1/(1 + (-x).exp())
        return self.last_output
    
    def backward(self, grad_wrt_output):
        grad_wrt_input = grad_wrt_output * (self.last_output * (1.0 - self.last_output)) # slide 10 of lecture 3.6
        return grad_wrt_input
    
    def params(self):
        return []