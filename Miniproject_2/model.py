from others.autoencoder import *
import torch
from torch import nn
from torch import optim

class Model():
    
    def __init__(self) -> None:
        ## instantiate model + optimizer + loss function + any other stuff you need
        pass
    
    
    def load_pretrained_model(self) -> None:
        ## This loads the parameters saved in bestmodel . pth into the model
        pass
        
    
    def train(self, train_input,train_target) -> None:
        # : train ̇input : tensor of size (N , C , H , W ) containing a noisy version of the images
        # : train ̇target : tensor of size (N , C , H , W ) containing another noisy version of the same images , which only differs from the input by their noise .
        pass
                
    
    def predict(self, test_input) -> torch.Tensor:
        # : test ̇input : tensor of size ( N1 , C , H , W ) that has to be denoised by the trained or the loaded network .
        pass
    