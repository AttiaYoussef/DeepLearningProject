import torch
torch.set_grad_enabled(False)

class Module(object):
    def forward(self, *input):
        pass
    def backward(self, *gradwrtoutput):
        pass
    def params(self):
        return []

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        if type(kernel) is int:
            self.window = torch.empty((kernel, kernel))
        else:
            self.window = torch.empty(kernel)
            
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