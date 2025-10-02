"""
Module containing the encoders.
"""
import numpy as np

import torch
import os
from torch import nn


# ALL encoders should be called Enccoder<Model>
def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Encoder{}".format(model_type))


class EncoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4 # !!!!!!!!!!!!!!!!!!!!!!
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1) # !!!!!!!!!!!!!!!!!!!!!!
        self.conv1 = nn.Conv2d(n_chan,         hid_channels, kernel_size, **cnn_kwargs) # * Change Oct/24
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs) # * Change Oct/24
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs) # * Change Oct/24


        self.num_conv = int(np.log2(self.img_size[1])) - 5 # computed to match the number of convolutions that were before
        self.num_conv += 0 # try to give it a bit more computational power # !!!  TO REMOVE
        self.list_convolutions = nn.ModuleList()
        for i in range(self.num_conv): # Define the needed number of convolutions
            self.list_convolutions.append(nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)) # * Change Oct/24
        self.batch1 = nn.BatchNorm2d(hid_channels) # * Change Oct/24
        self.max1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.batch2 = nn.BatchNorm2d(hid_channels) # * Change Oct/24
        self.max2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim // 2) # * Change Oct/31
#        self.lin2 = nn.Linear(hidden_dim // 1, hidden_dim // 2) # * Change Oct/31
        self.lin3 = nn.Linear(hidden_dim // 2, hidden_dim // 8) # * Change Oct/31
#        self.lin4 = nn.Linear(hidden_dim // 4, hidden_dim // 8) # * Change Oct/31

        # Dropout for improving generalization and regularization
        self.dropout = nn.Dropout(p=0.1) # * Change Oct/24


        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim // 8, self.latent_dim * 2)


    def forward(self, x):
        # print(x.shape)
        batch_size = x.size(0)
        # print(x[0,:,50,50])
        # print(x[0,:,100,100])
        # print(x[0,:,150,150])
        # print(x[0,:,200,200])
        # Convolutional layers with ReLu activations
        x = torch.nn.functional.leaky_relu(self.conv1(x), negative_slope=0.15) # * Change Oct/24
        #print(f'L1: {x.size()}')
        x = torch.nn.functional.leaky_relu(self.conv2(x), negative_slope=0.15) # * Change Oct/24
        #print(f'L2: {x.size()}')
        x = torch.nn.functional.leaky_relu(self.conv3(x), negative_slope=0.15) # * Change Oct/24
        #print(f'L3: {x.size()}')

        x = self.batch1(x)
        for conv_layer in self.list_convolutions:
            x = torch.nn.functional.leaky_relu(conv_layer(x), negative_slope=0.15) # * Change Oct/24
        x = self.batch2(x)

        # Fully connected layers with ReLu activations
        x = x.reshape((batch_size, -1)) # ! here there was a "view()" previously
        #print(f'view: {x.size()}')
        # x = self.dropout(torch.nn.functional.leaky_relu(self.lin1(x), negative_slope=0.15))
        x = torch.nn.functional.leaky_relu(self.lin1(x), negative_slope=0.15)
        #print(f'Lin1: {x.size()}')
#        x = torch.nn.functional.leaky_relu(self.lin2(x), negative_slope=0.15)
        #print(f'Lin2: {x.size()}')
        x = torch.nn.functional.leaky_relu(self.lin3(x), negative_slope=0.15) # * Change Oct/24
#        x = torch.nn.functional.leaky_relu(self.lin4(x), negative_slope=0.15) # * Change Oct/24

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)
        # print(f'Representation in Z: {mu}')
        return mu, logvar
