"""
Module containing the main VAE class.
"""
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

from disvae.utils.initialization import weights_init
from .encoders import get_encoder
from .decoders import get_decoder

MODELS = ["Burgess"]

def init_specific_model(model_type, img_size, latent_dim, bs=None, use_normals=False, skip=0):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    model_type = model_type.lower().capitalize()
    if model_type not in MODELS:
        err = "Unkown model_type={}. Possible values: {}"
        raise ValueError(err.format(model_type, MODELS))

    encoder = get_encoder(model_type)
    decoder = get_decoder(model_type)
    # print(f'Initializing VAE with Z{latent_dim}')
    model = VAE(img_size, encoder, decoder, latent_dim, bs, use_normals, skip)
    model.model_type = model_type  # store to help reloading
    return model


class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim, bs, use_normals=False, skip=0):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()
        
        # If the size has decimals, it is not a power of 2 -> bad
        if (np.log2(img_size[1]) % 1 != 0) or (np.log2(img_size[1]) % 1 != 0) or (img_size[1] != img_size[2]): 
            raise RuntimeError("{} sized images not supported. Please use images whose shapes are power of 2: (None, 256x256), (None, 512x512)...".format(img_size))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim)
        self.decoder = decoder(img_size, self.latent_dim, bs, use_normals, skip)

        self.reset_parameters()


    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x, n=None):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        n : torch.Tensor
            Batch of normals (if any). Same shape as x
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample, n)
        return reconstruct, latent_dist, latent_sample

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample
