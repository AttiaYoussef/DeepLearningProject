from Miniproject_1.others.autoencoder import *
import torch
from torch import nn
from torch import optim
import pickle


class Model():
    
    def __init__(self) -> None:
        # # instantiate model + optimizer + loss function + any other stuff you need
        
        if torch.cuda.is_available() :
            device = torch.device("cuda:0")
        else :
            device = torch.device("cpu")
        
        self.model = Noise2Noise(encoding_block_dropout = 0, decoding_block_dropout = 0)
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0015, betas=(0.9, 0.99), eps=1e-08, weight_decay=0)       
        self.loss = nn.HuberLoss(delta=2.0).to(device)
    
    
    def load_pretrained_model(self) -> None:
        ## This loads the parameters saved in bestmodel . pth into the model
        if torch.cuda.is_available() :
            device = torch.device("cuda:0")
        else :
            device = torch.device("cpu")
        self.model.load_state_dict(torch.load('Miniproject_1/bestmodel.pth', map_location=device))
        
    
    
    def train(self, train_input,train_target, num_epochs = 7) -> None:
        # : train ̇input : tensor of size (N , C , H , W ) containing a noisy version of the images
        # : train ̇target : tensor of size (N , C , H , W ) containing another noisy version of the same images , which only differs from the input by their noise .
        torch.set_grad_enabled(True)
        
        normalized_input = train_input / 255.0
        normalized_target = train_target / 255.0
        
        N = train_input.size(0)
        batch_size = 32
        for e in range(num_epochs):
            for index in range(0, N, batch_size):
                self.optimizer.zero_grad()
                train_data_minibatch = normalized_input[index:(index+batch_size)]
                train_target_minibatch = normalized_target[index:(index+batch_size)]
                for p in self.model.parameters():
                    p.retain_grad()
                self.loss(self.model(train_data_minibatch), train_target_minibatch).backward()
                self.optimizer.step()
                
    
    def predict(self, test_input) -> torch.Tensor:
        # : test ̇input : tensor of size ( N1 , C , H , W ) that has to be denoised by the trained or the loaded network .
        return torch.clamp(self.model(test_input  / 255.0), min = 0, max = 1) * 255.0
    