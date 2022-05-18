"""
This file contains all necessary functions and classes to implement the autoencoder described in the paper 
"Noise2Noise: Learning Image Restoration without Clean Data"
Link: https://arxiv.org/abs/1803.04189
"""

import torch
from torch import nn

"""
Describes an "encoding block" in the autoencoder; it's composed of a 3x3 convolutional layer, followed by either
a leaky relu or linear activation layer, and a 2x2 maxpool layer
"""

class EncodingBlock(nn.Module):
    
    def __init__(self, n_in, n_out, dropout = True, p = 0.2):
        super().__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = 3, padding = 'same', padding_mode = 'zeros')
        self.after = nn.Identity()
        if dropout:
            self.after = nn.Dropout(p = p)
        self.norm = nn.BatchNorm2d(n_out)
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.after(x)
        x = self.pool(x)
        
        return x
    
"""
Describes an "decoding block" in the autoencoder; it's composed of a 3x3 convolutional layer, followed by either
a leaky relu or linear activation layer, and a 2x2 upsample layer
"""
class DecodingBlock(nn.Module):
    
    def __init__(self, n_in, n_out, dropout = True, p = 0.2):
        super().__init__()
        
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = 3, padding = 'same', padding_mode = 'zeros')
        self.after = nn.Identity()
        if dropout:
            self.after = nn.Dropout(p = p)
        self.norm = nn.BatchNorm2d(n_out)
        self.up = nn.Upsample(scale_factor = 2,mode = 'nearest')
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.up(x)
        x = self.after(x)
        return x
    
"""
Implements the autoencoder
"""
class Noise2Noise(nn.Module):
    
    def __init__(self, encoding_block_n_in = 48, encoding_block_n_out = 48, decoding_block_n_in = 96,decoding_block_n_out = 96, deconv_block_n_in = 144 ,  n = 3, m = 3, encoding_block_dropout = 0.2, decoding_block_dropout = 0.2):    
        super().__init__()
        #### First layer
        self.conv0 = nn.Conv2d(n, encoding_block_n_in, kernel_size = 3, padding = 'same', padding_mode = 'zeros')
        #### ENCODING BLOCKS
        self.encoding1 = EncodingBlock(encoding_block_n_in, encoding_block_n_out, p = encoding_block_dropout)
        self.encoding2 = EncodingBlock(encoding_block_n_in, encoding_block_n_out, p = encoding_block_dropout)
        self.encoding3 = EncodingBlock(encoding_block_n_in, encoding_block_n_out, p = encoding_block_dropout)
        
        #### DECODING BLOCKS
        self.decoding1 = DecodingBlock(encoding_block_n_in,encoding_block_n_out, p = decoding_block_dropout)
        self.deconv1 = nn.Conv2d(decoding_block_n_in, decoding_block_n_out, kernel_size = 3, padding = 'same', padding_mode = 'zeros')
        self.decoding2 = DecodingBlock(decoding_block_n_in,decoding_block_n_out, p = decoding_block_dropout)
        self.deconv2 = nn.Conv2d(deconv_block_n_in, decoding_block_n_out, kernel_size = 3, padding = 'same', padding_mode = 'zeros')

        self.decoding3 = DecodingBlock(decoding_block_n_in,decoding_block_n_out, p = decoding_block_dropout)
        
        #### Last layers
        self.conv1 = nn.Conv2d(decoding_block_n_in + n, 64, kernel_size = 3, padding = 'same', padding_mode = 'zeros')
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 3, padding = 'same', padding_mode = 'zeros')
        self.conv3 = nn.Conv2d(32, m, kernel_size = 3, padding = 'same', padding_mode = 'zeros')
        
    def forward(self, x):
        input_ = x.detach().clone()
        n = m = 3
        x = self.conv0(x) #enc_conv0
        x = nn.LeakyReLU(0.1)(x)
        #### ENCODING PHASE
        pool1 = self.encoding1(x) #pool1
        pool2 = self.encoding2(pool1) #pool2
        pool3 = self.encoding3(pool2) #pool3
        
        
        #### DECODING PHASE
        
        upsample3 = self.decoding1(pool3)
        concat3 = torch.cat([upsample3,pool2], dim = 1)
        dec_conv3a = self.deconv1(concat3)
        dec_conv3a = nn.LeakyReLU(0.1)(dec_conv3a)
        
        upsample2 = self.decoding2(dec_conv3a)
        concat2 = torch.cat([upsample2,pool1], dim = 1)
        dec_conv2a = self.deconv2(concat2)
        dec_conv2a = nn.LeakyReLU(0.1)(dec_conv2a)
        
        
        upsample1 = self.decoding3(dec_conv2a)
        concat1 = torch.cat([upsample1,input_], dim = 1)
        
        
        #### Last phase
        r = self.conv1(concat1)
        r = nn.LeakyReLU(0.1)(r)
        
        r = self.conv2(r)
        r = nn.LeakyReLU(0.1)(r)
        
        r = self.conv3(r)
        return r