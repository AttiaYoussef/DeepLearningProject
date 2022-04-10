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
    
    def __init__(self, n_in, n_out):
        super.__init__()
        
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = 3, padding_mode = 'zeros')
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        
        return x
    
"""
Describes an "decoding block" in the autoencoder; it's composed of a 3x3 convolutional layer, followed by either
a leaky relu or linear activation layer, and a 2x2 upsample layer
"""
def DecodingBlock(nn.Module):
    
    def __init__(self, n_in, n_out):
        super.__init__()
        
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = 3, padding_mode = 'zeros') 
        self.up = nn.Upsample(size = None, scale_factor = None, mode = 'nearest')
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.up(x)
        return x
    
"""
Implements the autoencoder
"""
def Noise2Noise(nn.Module):
    
    def __init__(self, n = 3, m = 3):    
        
        #### ENCODING BLOCKS
        self.encoding1 = EncodingBlock(48, 48)
        self.encoding2 = EncodingBlock(48, 48)
        self.encoding3 = EncodingBlock(48, 48)
        self.encoding4 = EncodingBlock(48, 48)
        self.encoding5 = EncodingBlock(48, 48)
        
        #### DECODING BLOCKS
        self.decoding1 = DecodingBlock(48,48)
        self.decoding2 = DecodingBlock(96,96)
        self.decoding3 = DecodingBlock(96,96)
        self.decoding4 = DecodingBlock(96,96)
        self.decoding5 = DecodingBlock(96,96)
        
        #### Last layers
        self.conv1 = nn.Conv2d(96 + n, 64, kernel_size = 3, padding_mode = 'zeros')
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 3, padding_mode = 'zeros')
        self.conv3 = nn.Conv2d(32, m, kernel_size = 3, padding_mode = 'zeros')
        
    def forward(self, x):
        input_ = x.detach().clone()
        n = m = 3
        x = nn.Conv2d(n, 48, kernel_size = 3, padding_mode = 'zeros')(x) #enc_conv0
        x = nn.LeakyReLU(0.1)(x)
        
        #### ENCODING PHASE
        pool1 = self.encoding1(x) #pool1
        pool2 = self.encoding2(pool1) #pool2
        pool3 = self.encoding3(pool2) #pool3
        pool4 = self.encoding3(pool3) #pool4
        pool5 = self.encoding3(pool4) #pool5
        
        
        #### DECODING PHASE
        upsample5 = self.decoding1(pool5)
        concat5 = None
        dec_conv5a = nn.Conv2d(96, 96, kernel_size = 3, padding_mode = 'zeros')(concat5)
        dec_conv5a = nn.LeakyReLU(0.1)(dec_conv5a)
        
        upsample4 = self.decoding2(dec_conv5a)
        concat4 = None
        dec_conv4a = nn.Conv2d(144, 96, kernel_size = 3, padding_mode = 'zeros')(concat4)
        dec_conv4a = nn.LeakyReLU(0.1)(dec_conv4a)
        
        upsample3 = self.decoding2(dec_conv4a)
        concat3 = None
        dec_conv3a = nn.Conv2d(144, 96, kernel_size = 3, padding_mode = 'zeros')(concat3)
        dec_conv3a = nn.LeakyReLU(0.1)(dec_conv3a)
        
        upsample2 = self.decoding2(dec_conv3a)
        concat2 = None
        dec_conv2a = nn.Conv2d(144, 96, kernel_size = 3, padding_mode = 'zeros')(concat2)
        dec_conv2a = nn.LeakyReLU(0.1)(dec_conv2a)
        
        
        upsample1 = self.decoding2(dec_conv2a)
        concat1 = None
        
        
        #### Last phase
        r = self.conv1(concat1)
        r = nn.LeakyReLU(0.1)(r)
        
        r = self.conv2(r)
        r = nn.LeakyReLU(0.1)(r)
        
        r = self.conv3(r)
        return r