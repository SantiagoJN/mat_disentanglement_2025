import os
from math import ceil, floor

import imageio
from PIL import Image
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
from skimage import img_as_ubyte

from utils.datasets import get_background
from utils.viz_helpers import (read_loss_from_file, add_labels, make_grid_img,
                               sort_list_by_other, FPS_GIF, concatenate_pad,
                               plot_gaussians)
from utils.helpers import create_safe_directory
import copy

TRAIN_FILE = "train_losses.log"
DECIMAL_POINTS = 3
GIF_FILE = "training.gif"
PLOT_NAMES = dict(generate_samples="samples.png",
                  data_samples="data_samples.png",
                  reconstruct="reconstruct.png",
                  traversals="traversals.png",
                  reconstruct_traverse="reconstruct_traverse.png",
                  gif_traversals="posterior_traversals.gif",)


class Visualizer():
    def __init__(self, model, dataset, model_dir,
                 save_images=True,
                 loss_of_interest=None,
                 display_loss_per_dim=False,
                 max_traversal=0.475,  # corresponds to ~2 for standard normal
                 upsample_factor=1):
        """
        Visualizer is used to generate images of samples, reconstructions,
        latent traversals and so on of the trained model.

        Parameters
        ----------
        model : disvae.vae.VAE

        dataset : str
            Name of the dataset.

        model_dir : str
            The directory that the model is saved to and where the images will
            be stored.

        save_images : bool, optional
            Whether to save images or return a tensor.

        loss_of_interest : str, optional
            The loss type (as saved in the log file) to order the latent dimensions by and display.

        display_loss_per_dim : bool, optional
            if the loss should be included as text next to the corresponding latent dimension images.

        max_traversal: float, optional
            The maximum displacement induced by a latent traversal. Symmetrical
            traversals are assumed. If `m>=0.5` then uses absolute value traversal,
            if `m<0.5` uses a percentage of the distribution (quantile).
            E.g. for the prior the distribution is a standard normal so `m=0.45` c
            orresponds to an absolute value of `1.645` because `2m=90%%` of a
            standard normal is between `-1.645` and `1.645`. Note in the case
            of the posterior, the distribution is not standard normal anymore.

        upsample_factor : floar, optional
            Scale factor to upsample the size of the tensor
        """
        self.model = model
        self.device = next(self.model.parameters()).device
        self.latent_dim = self.model.latent_dim
        self.max_traversal = max_traversal
        self.save_images = save_images
        self.model_dir = model_dir
        self.dataset = dataset
        self.upsample_factor = upsample_factor
        if loss_of_interest is not None:
            self.losses = read_loss_from_file(os.path.join(self.model_dir, TRAIN_FILE),
                                              loss_of_interest)

    def _get_traversal_range(self, mean=0, std=1):
        """Return the corresponding traversal range in absolute terms."""
        max_traversal = self.max_traversal

        if max_traversal < 0.5:
            max_traversal = (1 - 2 * max_traversal) / 2  # from 0.45 to 0.05
            max_traversal = stats.norm.ppf(max_traversal, loc=mean, scale=std)  # from 0.05 to -1.645

        # symmetrical traversals
        # print(f'Max traversal: {max_traversal}')
        return ((-1 * max_traversal + mean), max_traversal+mean)

    def _traverse_line(self, idx, n_samples, data=None):
        """Return a (size, latent_size) latent sample, corresponding to a traversal
        of a latent variable indicated by idx.

        Parameters
        ----------
        idx : int
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and idx = 7, then the 7th dimension
            will be traversed while all others are fixed.

        n_samples : int
            Number of samples to generate.

        data : torch.Tensor or None, optional
            Data to use for computing the posterior. Shape (N, C, H, W). If
            `None` then use the mean of the prior (all zeros) for all other dimensions.
        """
        if data is None:
            # mean of prior for other dimensions
            samples = torch.zeros(n_samples, self.latent_dim)
            traversals = torch.linspace(*self._get_traversal_range(), steps=n_samples)

        else:
            if data.size(0) > 1:
                raise ValueError("Every value should be sampled from the same posterior, but {} datapoints given.".format(data.size(0)))

            with torch.no_grad():
                post_mean, post_logvar = self.model.encoder(data.to(self.device))
                # print(f'Post mean: {post_mean}')
                # print(f'Post logvar: {post_logvar}')
                #! Plot_gaussians
#                plot_gaussians(post_mean.cpu().numpy(),post_logvar.cpu().numpy(),sample_name="[1_1_cambridge_2k][sphere][UTIA-m076_fabric113]")
#                exit()
                
                samples = self.model.reparameterize(post_mean, post_logvar)
                samples = samples.cpu().repeat(n_samples, 1)
                post_mean_idx = post_mean.cpu()[0, idx]
                post_std_idx = torch.exp(post_logvar / 2).cpu()[0, idx]

            # travers from the gaussian of the posterior in case quantile
            traversals = torch.linspace(*self._get_traversal_range(mean=post_mean_idx,
                                                                   std=post_std_idx),
                                        steps=n_samples)
        
        # if idx == 3:
        #     print('-----------------traversals-----------------')
        #     print(traversals)

        #     print('-----------------post_mean-----------------')
        #     print(post_mean)

        #     print('-----------------post_mean_idx-----------------')
        #     print(post_mean_idx)

        #     print('-----------------samples (before)-----------------')
        #     print(samples)

        for i in range(n_samples):
            samples[i, idx] = traversals[i]

        # if idx == 3:
        #     print('-----------------samples (after)-----------------')
        #     print(samples.shape)
        #     print(samples)
        #     exit()

        return samples

    def _save_or_return(self, to_plot, size, filename, is_force_return=False):
        """Create plot and save or return it."""
        to_plot = F.interpolate(to_plot, scale_factor=self.upsample_factor)

        if size[0] * size[1] != to_plot.shape[0]:
            raise ValueError("Wrong size {} for datashape {}".format(size, to_plot.shape))

        # `nrow` is number of images PER row => number of col
        kwargs = dict(nrow=size[1], pad_value=(1 - get_background(self.dataset)))
        if self.save_images and not is_force_return:
            filename = os.path.join(self.model_dir, filename)
            save_image(to_plot, filename, **kwargs)
        else:
            return make_grid_img(to_plot, **kwargs)

    def _decode_latents(self, latent_samples):
        """Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = latent_samples.to(self.device)
        return self.model.decoder(latent_samples).cpu()

    def generate_samples(self, size=(8, 8)):
        """Plot generated samples from the prior and decoding.

        Parameters
        ----------
        size : tuple of ints, optional
            Size of the final grid.
        """
        prior_samples = torch.randn(size[0] * size[1], self.latent_dim)
        prior_samples = prior_samples
        generated = self._decode_latents(prior_samples)
        return self._save_or_return(generated.data, size, PLOT_NAMES["generate_samples"])

    def data_samples(self, data, size=(8, 8)):
        """Plot samples from the dataset

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints, optional
            Size of the final grid.
        """
        data = data[:size[0] * size[1], ...]
        return self._save_or_return(data, size, PLOT_NAMES["data_samples"])

    
    def plot_gaussians(self, data):
        """Method to plot the representations of the gaussians obtained with
            a certain sample

        Parameters
        ----------
        idx : int
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and idx = 7, then the 7th dimension
            will be traversed while all others are fixed.

        n_samples : int
            Number of samples to generate.

        data : torch.Tensor or None, optional
            Data to use for computing the posterior. Shape (N, C, H, W). If
            `None` then use the mean of the prior (all zeros) for all other dimensions.
        """
    
        if data.size(0) > 1:
            raise ValueError("Every value should be sampled from the same posterior, but {} datapoints given.".format(data.size(0)))

        with torch.no_grad():
            post_mean, post_logvar = self.model.encoder(data.to(self.device))
            filename=os.path.join(self.model_dir, f"gaussians_plot.png")
            plot_gaussians(post_mean.cpu().numpy(),post_logvar.cpu().numpy(),sample_name="[1_1_cambridge_2k][sphere][UTIA-m076_fabric113]",filename=filename)



    # Get the latent representation of some data
    def get_representation(self, data, results_path, experiment_name):
        # data = torch.from_numpy(data) # First convert from *np* to *torch*
        # with torch.no_grad():
        #     mean, variance = self.model.encoder(data.to(self.device))

        
        # Just like in run_PCA.py
        batch_size = 585 # ! Tune those two values properly
        num_batches = 16
        latent_dim = 20
        if(batch_size * num_batches != data.shape[0]):
            print(f'[ERROR] The product between the batch size ({batch_size}) '
                  f'and the number of batches ({num_batches}) must be equal to'
                  f' the number of samples! ({data.shape[0]})')
            print(f'[INFO] Try to make the number of batches large, so the individual'
                  ' batches can fit in the GPU.')
            exit()

        latent_mean = np.zeros((batch_size*num_batches,latent_dim), dtype=float) # Save space to store intermediate results
        latent_sample = np.zeros((batch_size*num_batches,latent_dim), dtype=float) # To store variances instead of just means
        for batch in range(num_batches): # Get the latent representation in batches
            print(f'    Getting representations of batch No.{batch}/{num_batches}')
            torch.cuda.empty_cache()
        # Check the shapes of the images and the labels
            batch_imgs = data[range(batch*batch_size, (batch+1)*batch_size),:,:,:] # Get a batch of 64 images
            batch_imgs = torch.from_numpy(batch_imgs) # Convert it to a torch tensor
            batch_imgs = batch_imgs.to(self.device) # Send it to the device
            batch_imgs = batch_imgs.float() # Converting it into float because otherwise it outputs errors

            # Introduce every image to the encoder and check that the output is 20D
            with torch.no_grad():
                mean, variance = self.model.encoder(batch_imgs) # Finally get the latent representation

            mean = mean.to(torch.device("cpu")) # Send it back to the cpu
            mean = mean.detach().numpy()
            variance = variance.to(torch.device("cpu")) # Send it back to the cpu
            variance = variance.detach().numpy()
            
            latent_mean[batch*batch_size:(batch+1)*batch_size, :] = mean # Save the intermediate result
            latent_sample[batch*batch_size:(batch+1)*batch_size, :] = variance
        
        mean_path = f'{results_path}/mean'
        print(f'Saving means to {mean_path}')
        np.save(mean_path, latent_mean)
        variance_path = f'{results_path}/variance'
        print(f'Saving variances to {variance_path}')
        np.save(variance_path, latent_sample)
        f = open(f'{results_path}/name', "w")
        f.write(experiment_name)

        print(f'Mean shape: {latent_mean.shape}, variance shape: {latent_sample.shape}')
        
        # return mean, variance

    def reconstruct(self, data, size=(8, 8), is_original=True, is_force_return=False):
        """Generate reconstructions of data through the model.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints, optional
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even when `is_original`, so that upper
            half contains true data and bottom half contains reconstructions.contains

        is_original : bool, optional
            Whether to exclude the original plots.

        is_force_return : bool, optional
            Force returning instead of saving the image.
        """

        # create_safe_directory(os.path.join(self.model_dir,"source_samples"))
        # create_safe_directory(os.path.join(self.model_dir,"reconstructed_samples"))
        # for image in range(len(data)):
        #     print(f'Image {image}')
        #     plt.imshow(np.swapaxes(np.swapaxes(data[image].detach().numpy(),0,2),0,1))
        #     plt.show()
            # filename = os.path.join(self.model_dir, f"source_samples/sample_{image}.png")
            # imageio.imwrite(filename, np.swapaxes(np.swapaxes(data[image].detach().numpy(),0,2),0,1))

        if is_original:
            if size[0] % 2 != 0:
                raise ValueError("Should be even number of rows when showing originals not {}".format(size[0]))
            n_samples = size[0] // 2 * size[1]
        else:
            n_samples = size[0] * size[1]

        with torch.no_grad():
            originals = data.to(self.device)[:n_samples, ...]
            recs, _, _ = self.model(originals)

        originals = originals.cpu()
        recs = recs.view(-1, *self.model.img_size).cpu()

        # for image in range(5):
            # plt.imshow(np.swapaxes(np.swapaxes(recs[image].detach().numpy(),0,2),0,1))
            # plt.show()
            # filename = os.path.join(self.model_dir, f"reconstructed_samples/sample_{image}.png")
            # imageio.imwrite(filename, np.swapaxes(np.swapaxes(recs[image].detach().numpy(),0,2),0,1))

        to_plot = torch.cat([originals, recs]) if is_original else recs
        return self._save_or_return(to_plot, size, PLOT_NAMES["reconstruct"],
                                    is_force_return=is_force_return)

    def traversals(self,
                   data=None,
                   is_reorder_latents=False,
                   n_per_latent=8,
                   n_latents=None,
                   is_force_return=False,
                   prefix="",
                   number_traversal=0):
        # print(f'########Data shape is {data.shape}')
        # plt.imshow(data.detach().numpy().reshape(-1,64))
        # plt.show()
        # plt.waitforbuttonpress()
        # os.exit()
        if data != None: # Do it only when using main_viz, not during training
            # data_print = data.detach().numpy().reshape(-1,64)
            data_print = copy.deepcopy(data) # !Make a deep copy so we don't mess up with the data
            data_print = data_print.detach().numpy().squeeze()
            data_print = np.swapaxes(data_print, 0, 2)
            data_print = np.swapaxes(data_print, 0, 1)
            data_print *= 255.0
            data_print = data_print.astype(np.uint8)
            filename = os.path.join(self.model_dir, f"{prefix}traversals_source_{number_traversal}.png")
            # https://github.com/zhixuhao/unet/issues/125
            # imageio.imwrite('#1.jpg', np.zeros([512, 512, 3], dtype=np.uint8))
            # img_as_ubyte(sh)
            imageio.imwrite(filename, data_print) # Save the image that will be used for traversals
        """Plot traverse through all latent dimensions (prior or posterior) one
        by one and plots a grid of images where each row corresponds to a latent
        traversal of one latent dimension.

        Parameters
        ----------
        data : bool, optional
            Data to use for computing the latent posterior. If `None` traverses
            the prior.

        n_per_latent : int, optional
            The number of points to include in the traversal of a latent dimension.
            I.e. number of columns.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        is_reorder_latents : bool, optional
            If the latent dimensions should be reordered or not

        is_force_return : bool, optional
            Force returning instead of saving the image.
        """
        n_latents = n_latents if n_latents is not None else self.model.latent_dim
        latent_samples = [self._traverse_line(dim, n_per_latent, data=data)
                          for dim in range(self.latent_dim)]
        #print(f'Shape of latent_samples: {latent_samples.size}')
        #print(f'latent_dim: {self.latent_dim}')
        decoded_traversal = self._decode_latents(torch.cat(latent_samples, dim=0))


        if is_reorder_latents:
            n_images, *other_shape = decoded_traversal.size()
            n_rows = n_images // n_per_latent
            decoded_traversal = decoded_traversal.reshape(n_rows, n_per_latent, *other_shape) #!
            # self.losses[36] = 0.001
            # self.losses[37] = 0.002
            # self.losses[38] = 0.003
            # self.losses[39] = 0.004

            # print(f'Decoded_traversal: {decoded_traversal.size()}, n_images: {n_images}, other_shape: {other_shape}')
            # print(f'Self.losses with length {len(self.losses)}: {self.losses}')
            # exit()
            decoded_traversal = sort_list_by_other(decoded_traversal, self.losses)
            decoded_traversal = torch.stack(decoded_traversal, dim=0)
            decoded_traversal = decoded_traversal.reshape(n_images, *other_shape)
        
        # sh = decoded_traversal
        # plt.imshow(sh.detach().numpy().reshape(-1,32))
        # plt.show()
        # plt.waitforbuttonpress()
        sh = decoded_traversal 
        sh = sh.detach().numpy().reshape(-1,64)
        #sh = sh.astype(np.uint8)
        filename = os.path.join(self.model_dir, "decoded_traversal.png")
        imageio.imwrite(filename, img_as_ubyte(sh)) # Save the image that will be used for traversals

        # sh = decoded_traversal
        # plt.imshow(sh.detach().numpy().reshape(-1,64))
        # plt.show()
        # plt.waitforbuttonpress()
        decoded_traversal = decoded_traversal[range(n_per_latent * n_latents), ...]

        

        size = (n_latents, n_per_latent)
        sampling_type = "prior" if data is None else "posterior"
        filename = "{}_{}".format(sampling_type, PLOT_NAMES["traversals"])

        return self._save_or_return(decoded_traversal.data, size, filename,
                                    is_force_return=is_force_return)

    def reconstruct_traverse(self, data,
                             is_posterior=True,
                             n_per_latent=8,
                             n_latents=None,
                             is_show_text=False):
        """
        Creates a figure whith first row for original images, second are
        reconstructions, rest are traversals (prior or posterior) of the latent
        dimensions.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        n_per_latent : int, optional
            The number of points to include in the traversal of a latent dimension.
            I.e. number of columns.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        is_posterior : bool, optional
            Whether to sample from the posterior.

        is_show_text : bool, optional
            Whether the KL values next to the traversal rows.
        """
        n_latents = n_latents if n_latents is not None else self.model.latent_dim

        reconstructions = self.reconstruct(data[:2 * n_per_latent, ...],
                                           size=(2, n_per_latent),
                                           is_force_return=True)
        traversals = self.traversals(data=data[0:1, ...] if is_posterior else None,
                                     is_reorder_latents=True,
                                     n_per_latent=n_per_latent,
                                     n_latents=n_latents,
                                     is_force_return=True)

        concatenated = np.concatenate((reconstructions, traversals), axis=0)
        concatenated = Image.fromarray(concatenated)

        if is_show_text:
            losses = sorted(self.losses, reverse=True)[:n_latents]
            labels = ['orig', 'recon'] + ["KL={:.4f}".format(l) for l in losses]
            concatenated = add_labels(concatenated, labels)

        filename = os.path.join(self.model_dir, PLOT_NAMES["reconstruct_traverse"])
        concatenated.save(filename)

    def gif_traversals(self, data, n_latents=None, n_per_gif=15, freq=0, individuals=False):
        """Generates a grid of gifs of latent posterior traversals where the rows
        are the latent dimensions and the columns are random images.

        Parameters
        ----------
        data : bool
            Data to use for computing the latent posteriors. The number of datapoint
            (batchsize) will determine the number of columns of the grid.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        n_per_gif : int, optional
            Number of images per gif (number of traversals)
        """

        posteriors_indiv_path = os.path.join(self.model_dir, "posteriors_indiv")
        if freq != 0:
            create_safe_directory(posteriors_indiv_path)
            
        gif_samples = os.path.join(self.model_dir, "gif_samples")
        create_safe_directory(gif_samples)

        n = 0 # Counter for storing the individual plots

        n_images, _, _, width_col = data.shape
        print(f'{n_images} IMAGES')
        width_col = int(width_col * self.upsample_factor)
        all_cols = [[] for c in range(n_per_gif)]
        for i in range(n_images):
            grid = self.traversals(data=data[i:i + 1, ...], is_reorder_latents=True,
                                   n_per_latent=n_per_gif, n_latents=n_latents,
                                   is_force_return=True, prefix="gif_samples/", number_traversal=i)
            # print(type(grid))
            # plt.figure(1)
            # a = plt.imshow((data[i:i + 1, ...]).reshape(32,32))
            # plt.figure(2)
            # b = plt.imshow(grid)
            # plt.show()
            # plt.waitforbuttonpress()

            if freq != 0: # We will need to manage the storing of the plots
                if n % freq == 0: # Time to store the current grid
                    filename = os.path.join(posteriors_indiv_path, f'post_{n}.jpeg')
                    imageio.imwrite(filename, grid)
                n += 1 # Update the counter

            height, width, c = grid.shape
            padding_width = (width - width_col * n_per_gif) // (n_per_gif + 1)

            # split the grids into a list of column images (and removes padding)
            for j in range(n_per_gif):
                all_cols[j].append(grid[:, [(j + 1) * padding_width + j * width_col + i
                                            for i in range(width_col)], :])

        pad_values = (1 - get_background(self.dataset)) * 255
        all_cols = [concatenate_pad(cols, pad_size=2, pad_values=pad_values, axis=1)
                    for cols in all_cols]
        
        # print(f'Type of all cols: {type(all_cols)}')
        # print(f'Length of all cols: {len(all_cols)}')
        # print(f'Type of the elements of all cols: {type(all_cols[0])}')
        gif_source = os.path.join(self.model_dir, "gif_source")
        create_safe_directory(gif_source)
        for image in range(len(all_cols)):
            filename = os.path.join(gif_source, f'frame_{image}.jpeg')
            imageio.imwrite(filename, all_cols[image])

        filename = os.path.join(self.model_dir, PLOT_NAMES["gif_traversals"])
        imageio.mimsave(filename, all_cols, fps=FPS_GIF)
        # gif1 = "gif_wu"
        # gif2 = "gif_nq"
        # imageio.mimsave(f'{os.path.join(self.model_dir, gif1)}.gif', all_cols, fps=FPS_GIF, quantizer='wu')
        # imageio.mimsave(f'{os.path.join(self.model_dir, gif2)}.gif', all_cols, fps=FPS_GIF, quantizer='nq')


class GifTraversalsTraining:
    """Creates a Gif of traversals by generating an image at every training epoch.

    Parameters
    ----------
    model : disvae.vae.VAE

    dataset : str
        Name of the dataset.

    model_dir : str
        The directory that the model is saved to and where the images will
        be stored.

    is_reorder_latents : bool, optional
        If the latent dimensions should be reordered or not

    n_per_latent : int, optional
        The number of points to include in the traversal of a latent dimension.
        I.e. number of columns.

    n_latents : int, optional
        The number of latent dimensions to display. I.e. number of rows. If `None`
        uses all latents.

    kwargs:
        Additional arguments to `Visualizer`
    """

    def __init__(self, model, dataset, model_dir,
                 is_reorder_latents=False,
                 n_per_latent=10,
                 n_latents=None,
                 store_freq=0,
                 **kwargs):
        self.save_filename = os.path.join(model_dir, GIF_FILE)
        self.visualizer = Visualizer(model, dataset, model_dir,
                                     save_images=False, **kwargs)

        self.images = []
        self.is_reorder_latents = is_reorder_latents
        self.n_per_latent = n_per_latent
        self.n_latents = n_latents if n_latents is not None else model.latent_dim

        self.freq = store_freq
        os.mkdir(f'{self.save_filename[:-12]}/traversals')
        self.n = 0 # Identifier of the next isolated traversal image to be stored

    def __call__(self):
        """Generate the next gif image. Should be called after each epoch."""
        cached_training = self.visualizer.model.training
        self.visualizer.model.eval()
        img_grid = self.visualizer.traversals(data=None,  # GIF from prior
                                              is_reorder_latents=self.is_reorder_latents,
                                              n_per_latent=self.n_per_latent,
                                              n_latents=self.n_latents)
        self.images.append(img_grid)

        if self.freq != 0: # If storing individual traversals is activated
            if self.n % self.freq == 0: # And it is time to store it
                filename = f'{self.save_filename[:-12]}/traversals/tra_{self.n}.jpeg'
                imageio.imwrite(filename, img_grid) # Save the last image
        self.n = self.n + 1 # Update the counter for the next identifier


        if cached_training:
            self.visualizer.model.train()

    def save_reset(self):
        """Saves the GIF and resets the list of images. Call at the end of training."""
        imageio.mimsave(self.save_filename, self.images, fps=FPS_GIF)
        self.images = []
