from others.autoencoder import *
import torch
from torch import nn
from torch import optim

class Model():
    
    def __init__(self) -> None:
        # # instantiate model + optimizer + loss function + any other stuff you need
        
        self.model = Noise2Noise()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-08, weight_decay=0)        
        
        #L2 loss: appropriate for Gaussian, multiplicative bernoulli, poisson noise
        #L0 loss: appropriate for random valued impulse noise
        self.loss = nn.MSELoss()
    
    
    def load_pretrained_model(self) -> None:
        ## This loads the parameters saved in bestmodel . pth into the model
        self.model.load_state_dict(torch.load('bestmodel.pth'))
        
    
    
    def train(self, train_input,train_target) -> None:
        # : train ̇input : tensor of size (N , C , H , W ) containing a noisy version of the images
        # : train ̇target : tensor of size (N , C , H , W ) containing another noisy version of the same images , which only differs from the input by their noise .
        epochs = 7
        N = train_input.size(0)
        batch_size = 32
        for e in range(epochs):
            for index in range(0, N, batch_size):
                self.optimizer.zero_grad()
                train_data_minibatch = train_input[index:(index+batch_size)]
                train_target_minibatch = train_target[index:(index+batch_size)]
                self.loss(self.model(train_data_minibatch), train_target_minibatch).backward()
                self.optimizer.step()
                
    
    def predict(self, test_input) -> None:
        # : test ̇input : tensor of size ( N1 , C , H , W ) that has to be denoised by the trained or the loaded network .
        return self.model(test_input)
    