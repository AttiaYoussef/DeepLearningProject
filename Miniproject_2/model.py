import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import math
import random
import pickle
from pathlib import Path
from .others.modules import *

class Model():
    
    def __init__(self) -> None:
        ## instantiate model + optimizer + loss function + any other stuff you need
        kernel = 3
        stride = 2
        
        self.model = Sequential(
            Conv2d(3,48,kernel,stride),
            ReLU(),
            Conv2d(48,48,kernel,stride),
            ReLU(),
            Conv2d(48,48,kernel,stride = 1, padding = 2),
            NNUpsampling(scale_factor = 2),
            Conv2d(48,24,kernel,stride = 1,),
            ReLU(),
            Conv2d(24,24,kernel,stride = 1, padding = 2),
            NNUpsampling(scale_factor = 2),
            Conv2d(24,24,kernel,stride = 1,),
            Conv2d(24,3,kernel,stride = 1,),
            Sigmoid()
        )
        
        self.optimizer = SGD(self.model.params(), lr = 0.15)
        self.loss = MSE()
    
    
    def load_pretrained_model(self) -> None:
        ## This loads the parameters saved in bestmodel . pth into the model
        
        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path, 'rb') as fs :
            self.model = pickle.load(fs)
    
    def train(self, train_input,train_target, num_epochs=15) -> None:
        # : train ̇input : tensor of size (N , C , H, W ) containing a noisy version of the images
        # : train ̇target : tensor of size (N , C , H , W ) containing another noisy version of the same images , which only differs from the input by their noise .
        torch.set_grad_enabled(False)
        
        batch_size = 32
        
        normalized_input = train_input / 255.0
        normalized_target = train_target / 255.0
        
        for i in range(num_epochs):
            for b in range(0, len(normalized_input), batch_size):
                self.optimizer.zero_grad()
                data = normalized_input[b:b+batch_size]
                target = normalized_target[b:b+batch_size]
                output=self.loss(self.model(data),target)
                self.model.backward(self.loss.backward())
                self.optimizer.step()
                
    def predict(self, test_input) -> torch.Tensor:
        # : test ̇input : tensor of size ( N1 , C , H , W ) that has to be denoised by the trained or the loaded network .
        return torch.clamp(self.model(test_input  / 255.0), min = 0, max = 1) * 255.0