"""
Module containing the decoders.
"""
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt


# ALL decoders should be called Decoder<Model>
def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Decoder{}".format(model_type))


class DecoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10,
                 bs=None, use_normals=False, skip=0):
        r"""Decoder of the model proposed in [1].

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
        super(DecoderBurgess, self).__init__()

        self.use_normals = use_normals
        # Layer parameters
        hid_channels = 64  # * Change Oct/24: 32 -> 64

        if self.use_normals:
            conv_hid_channels = hid_channels + 3 # The 3 channels of our normal map appended to the deconvolutions
            conv_hid_channels_NO_normals = hid_channels
        else:
            conv_hid_channels = hid_channels
            conv_hid_channels_NO_normals = hid_channels
        self.skip_first_conv = skip # Not using normal maps for the first n deconvolutions, to avoid including meaningless info (too blurry maps)
        print(f'Defining decoder with skip_first_conv = {self.skip_first_conv}')
        kernel_size = 4 # !!!!!!!!!!!!!!!!!!!!!!
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim // 8) # * Change Oct/31
        self.lin2 = nn.Linear(hidden_dim // 8, hidden_dim // 4) # * Change Oct/31
        self.lin3 = nn.Linear(hidden_dim // 4, hidden_dim // 2) # * Change Oct/24
        self.lin4 = nn.Linear(hidden_dim // 2, hidden_dim // 1) # * Change Oct/24
        self.lin5 = nn.Linear(hidden_dim // 1, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1) # !!!!!!!!!!!!!!!!!!!!!!

        self.num_conv = int(np.log2(self.img_size[1])) - 5 # computed to match the number of convolutions that were before
        self.num_conv += 0 # try to give it a bit more computational power # !!!!!!!!!!!!!!!!!!!!!!

        if self.skip_first_conv >= self.num_conv:
            print(f"[ERROR] skip_first_conv should be smaller than num_conv ({self.num_conv} > {self.skip_first_conv}). Skipping the first {self.num_conv} normal maps, but keeping the remaining 3 layers")

        self.list_convolutions = nn.ModuleList()
        for i in range(self.num_conv): # Define the needed number of convolutions
            if i >= self.skip_first_conv: # Include normals
                self.list_convolutions.append(nn.ConvTranspose2d(conv_hid_channels, hid_channels, kernel_size, **cnn_kwargs))
            else: # don't
                self.list_convolutions.append(nn.ConvTranspose2d(conv_hid_channels_NO_normals, hid_channels, kernel_size, **cnn_kwargs))
        
        #* Dim entrada=35, dim salida=32. Así, a la entrada de cada capa se le puede concatenar el mapa de normales redimensionado
        self.convT1 = nn.ConvTranspose2d(conv_hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(conv_hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(conv_hid_channels, n_chan, kernel_size, **cnn_kwargs)

        self.last_conv1 = nn.Conv2d(in_channels=n_chan + 3*use_normals, out_channels=n_chan, kernel_size=1)
        # self.last_conv2 = nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=1)

        self.batch_size = bs
        
    
    def adapt_and_cat(self, n, x): #! Keep an eye on this for optimization of the training process --> difficult since we have arbitrary normals..
        if not self.use_normals:
            return x
        # print(f'## Adapting normals to the shape {x.shape}')
        # print(f'## Normals are of type {type(n)} and shape {n.shape}')
        # exit()
        # print(f'Interpolating to shape {x.shape[2]}')
        n_resized = F.interpolate(n, size=x.shape[2]) # https://discuss.pytorch.org/t/resize-tensor-without-converting-to-pil-image/52401
        new_x = (torch.cat([x, n_resized], dim=1))
        # plt.imshow(np.swapaxes(np.swapaxes(n[0].cpu().numpy(),0,2),0,1))
        # plt.show()
        # plt.imshow(np.swapaxes(np.swapaxes(n_resized[0].cpu().numpy(),0,2),0,1))
        # plt.show()
        return new_x

    def forward(self, z, n=None):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.nn.functional.leaky_relu(self.lin1(z), negative_slope=0.15) # * Change Oct/24
        x = torch.nn.functional.leaky_relu(self.lin2(x), negative_slope=0.15) # * Change Oct/24
        x = torch.nn.functional.leaky_relu(self.lin3(x), negative_slope=0.15) # * Change Oct/24
        x = torch.nn.functional.leaky_relu(self.lin4(x), negative_slope=0.15) # * Change Oct/24
        x = torch.nn.functional.leaky_relu(self.lin5(x), negative_slope=0.15) # * Change Oct/24
        x = x.view(batch_size, *self.reshape)

        for i, conv_layer in enumerate(self.list_convolutions):
            if i >= self.skip_first_conv: # Include normals
                x = self.adapt_and_cat(n, x) # Concatenate normal map at right dimensions
            x = torch.nn.functional.leaky_relu(conv_layer(x), negative_slope=0.15) # * Change Oct/24

        x = self.adapt_and_cat(n, x) # Concatenate normal map at right dimensions
        x = torch.nn.functional.leaky_relu(self.convT1(x), negative_slope=0.15) # * Change Oct/24

        x = self.adapt_and_cat(n, x) # Concatenate normal map at right dimensions
        x = torch.nn.functional.leaky_relu(self.convT2(x), negative_slope=0.15) # * Change Oct/24
        
        # Sigmoid activation for final conv layer
        x = self.adapt_and_cat(n, x) # Concatenate normal map at right dimensions

        # * new implementation
        x = self.convT3(x) # ! APPLYING RELU CAN MAKE US LEARN LESS DIMENSIONS IN Z :(
        x = self.adapt_and_cat(n,x)
        x = torch.sigmoid(self.last_conv1(x))
        # * end of new implementation

        # * old implementation
        # x = torch.sigmoid(self.convT3(x))
        # * end of old implementation

        return x
