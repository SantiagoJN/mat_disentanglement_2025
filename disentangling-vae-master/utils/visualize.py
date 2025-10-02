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
from torchvision import transforms

import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.io import imread
from concurrent.futures import ThreadPoolExecutor
import cv2

from utils.datasets import get_background
from utils.viz_helpers import (read_loss_from_file, add_labels, make_grid_img,
                               sort_list_by_other, FPS_GIF, concatenate_pad,
                               plot_gaussians)
from utils.helpers import create_safe_directory
import copy

from skimage.metrics import structural_similarity
import lpips
from math import log10, sqrt 

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
                 upsample_factor=1,
                 use_normals=False,
                 normal_img="uninitialized"):
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
        self.warned = False # to just prompt the warning once...
        self.device = next(self.model.parameters()).device
        self.latent_dim = self.model.latent_dim
        self.max_traversal = max_traversal
        self.save_images = save_images
        self.model_dir = model_dir
        self.dataset = dataset
        self.upsample_factor = upsample_factor
        self.use_normals = use_normals
        self.normal_img = normal_img
        if loss_of_interest is not None:
            self.losses = read_loss_from_file(os.path.join(self.model_dir, TRAIN_FILE),
                                              loss_of_interest)

    def _get_traversal_range(self, mean=0, std=1):
        """Return the corresponding traversal range in absolute terms."""
        max_traversal = self.max_traversal

        if max_traversal < 0.5:
            max_traversal = (1 - 2 * max_traversal) / 2
            max_traversal = stats.norm.ppf(max_traversal, loc=mean, scale=std)

        # symmetrical traversals
        return ((-1 * max_traversal + mean), max_traversal+mean)


    # https://stackoverflow.com/questions/53345583/python-numpy-exponentially-spaced-samples-between-a-start-and-end-value
    def powspace(self, start, stop, power, num):
        print(f'power({start},{1/float(power)})')
        start = np.power(start, 1/float(power))
        stop = np.power(stop, 1/float(power))
        print(f'{start} - {stop}')
        return np.power( np.linspace(start, stop, num=num), power) 

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

        else: # posteriors
            if data.size(0) > 1:
                raise ValueError("Every value should be sampled from the same posterior, but {} datapoints given.".format(data.size(0)))

            with torch.no_grad():
                post_mean, post_logvar = self.model.encoder(data.to(self.device))
                
                samples = self.model.reparameterize(post_mean, post_logvar)
                samples = samples.cpu().repeat(n_samples, 1)
                post_mean_idx = post_mean.cpu()[0, idx]
                post_std_idx = torch.exp(post_logvar / 2).cpu()[0, idx]

            # travers from the gaussian of the posterior in case quantile
            traversals = torch.linspace(*self._get_traversal_range(mean=post_mean_idx,
                                                                   std=post_std_idx),
                                        steps=n_samples)

        for i in range(n_samples):
            samples[i, idx] = traversals[i]

        return samples

    def _save_or_return(self, to_plot, size, filename, is_force_return=False):
        """Create plot and save or return it."""
        to_plot = F.interpolate(to_plot, scale_factor=self.upsample_factor)

        if size[0] * size[1] != to_plot.shape[0]:
            raise ValueError("Wrong size {} for datashape {}".format(size, to_plot.shape))

        # `nrow` is number of images PER row => number of col
        kwargs = dict(nrow=size[1], pad_value=(1 - get_background(self.dataset)))
        to_plot = to_plot[:,:3,:,:] # just in case normal map is concatenated to the input
        if self.save_images and not is_force_return:
            filename = os.path.join(self.model_dir, filename)
            save_image(to_plot, filename, **kwargs)
        else:
            return make_grid_img(to_plot, **kwargs)

    def _get_normals(self, batch_size):
        normal_path = f"data/new_dataset_v2.2/normals/{self.normal_img}.png"
        if not self.warned:
            print(f'[WARNING] Using normals located at {normal_path}')
            self.warned = True
        normal = imread(normal_path)
        normal = normal[:,:,:3] # Remove the alpha channel from the png image
        t = transforms.ToTensor()
        normal = t(normal) #? For some reason, this to_tensor provides a shape of ch*w*h, instead of w*h*ch that gives the torch.from_numpy
        normal = normal.unsqueeze(0).repeat(batch_size, 1, 1, 1) # Repeat it to mimic a batch of normals
        normal = normal.to('cuda:0')

        return normal

    def _decode_latents(self, latent_samples):
        """Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = latent_samples.to(self.device)
        if self.use_normals:
            n = self._get_normals(latent_samples.shape[0])
            return self.model.decoder(latent_samples, n).cpu()
        else: # Not using normals in the decoder!
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

    def PSNR(self, original, compressed): 
        mse = np.mean((original - compressed) ** 2) 
        if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                    # Therefore PSNR have no importance. 
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse)) 
        return psnr 

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

        if is_original:
            if size[0] % 2 != 0:
                raise ValueError("Should be even number of rows when showing originals not {}".format(size[0]))
            n_samples = size[0] // 2 * size[1]
        else:
            n_samples = size[0] * size[1]

        with torch.no_grad():
            originals = data.to(self.device)[:n_samples, ...]
            if self.use_normals:
                n = self._get_normals(originals.shape[0])
                recs, _, _ = self.model(originals, n)
            else:
                recs, _, _ = self.model(originals)

        originals = originals.cpu()
        recs = recs.view(-1, *self.model.img_size).cpu()
      
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
        if data != None: # Do it only when using main_viz, not during training
            data_print = copy.deepcopy(data) # Make a deep copy so we don't mess up with the data
            data_print = data_print.detach().numpy().squeeze()
            data_print = np.swapaxes(data_print, 0, 2)
            data_print = np.swapaxes(data_print, 0, 1)
            data_print *= 255.0
            data_print = data_print.astype(np.uint8)
            data_print = data_print[:,:,:3]
            filename = os.path.join(self.model_dir, f"{prefix}traversals_source_{number_traversal}.png")
            # https://github.com/zhixuhao/unet/issues/125
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
        decoded_traversal = self._decode_latents(torch.cat(latent_samples, dim=0))


        if is_reorder_latents:
            n_images, *other_shape = decoded_traversal.size()
            n_rows = n_images // n_per_latent
            decoded_traversal = decoded_traversal.reshape(n_rows, n_per_latent, *other_shape) #!
            decoded_traversal = sort_list_by_other(decoded_traversal, self.losses)
            decoded_traversal = torch.stack(decoded_traversal, dim=0)
            decoded_traversal = decoded_traversal.reshape(n_images, *other_shape)

        sh = decoded_traversal 
        sh = sh.detach().numpy().reshape(-1,64)

        decoded_traversal = decoded_traversal[range(n_per_latent * n_latents), ...]

        

        size = (n_latents, n_per_latent)
        sampling_type = "prior" if data is None else "posterior"
        if is_reorder_latents == False:
            filename = "unordered_latents.png"
        else:
            filename = "{}_{}".format(sampling_type, PLOT_NAMES["traversals"])

        return self._save_or_return(decoded_traversal.data, size, filename,
                                    is_force_return=is_force_return)

    def grid(self,
                   n_per_latent=8,
                   is_force_return=False):
        """Plot to show a 2D grid of the combinations between different dimensions in the latent space
        """
        torch.set_printoptions(sci_mode=False)
        directory_name = "grids"
        directory_path = os.path.join(self.model_dir, directory_name)
        create_safe_directory(directory_path)

        for d1 in range(0, self.latent_dim):
            for d2 in range(0, self.latent_dim): # there will be redundant grids, just for the sake of completeness (maybe we want some specific orientation)
                n_latents = self.model.latent_dim
                latent_samples = [self._traverse_line(dim, n_per_latent) for dim in range(self.latent_dim)]
                latent_dim1 = self._traverse_line(d1, n_per_latent)
                latent_dim2 = self._traverse_line(d2, n_per_latent)
                # print(f'Latent samples: {latent_samples}, len of list: {len(latent_samples)}, shape of tensor: {latent_samples[0].shape}')
                
                combinations = []
                for t1 in latent_dim1:
                    for t2 in latent_dim2:
                        combinations.append(t1 + t2)
                
                combinations_array = torch.stack(combinations)
                decoded_traversal = self._decode_latents(combinations_array)
                
                sh = decoded_traversal 
                sh = sh.detach().numpy().reshape(-1,64)
                decoded_traversal = decoded_traversal[range(n_per_latent * n_per_latent), ...]

                size = (n_per_latent, n_per_latent)
                filename = f"{directory_name}/grid_{d1}_{d2}.png"
                self._save_or_return(decoded_traversal.data, size, filename,
                                        is_force_return=is_force_return)
                
    # Function to show the most similar images of a reference one (not used in the final paper)
    def similarity(self, data):
        from skimage.metrics import structural_similarity as ssim
        def compute_ssim(imageA, imageB):
            """Compute the average SSIM between two images across RGB channels."""
            ssim_scores = []
            for i in range(3):  # Loop over each channel (B, G, R)
                score, _ = ssim(imageA[:,:,i], imageB[:,:,i], full=True)
                ssim_scores.append(score)
            return np.mean(ssim_scores)

        def load_image(image_path):
            """Load an image from a file."""
            return cv2.imread(image_path)

        def process_image(image_path, reference_image):
            """Process a single image to compute its similarity to the reference image using SSIM."""
            image = load_image(image_path)
            similarity = compute_ssim(reference_image, image)
            return (similarity, image_path)

        def find_most_similar_images(reference_img, dataset_folder, top_n=4):
            """Find the most similar images to the reference image in the dataset folder using SSIM."""
            image_paths = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            # Use ThreadPoolExecutor to parallelize the processing of images
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_image, image_path, reference_img) for image_path in image_paths]
                similarities = [future.result() for future in futures]
            
            # Sort by similarity (highest SSIM first)
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Return the paths of the top_n most similar images
            return [path for _, path in similarities[:top_n]]

        def reconstructions(data):
            # put each pixel in [0.,1.] and reshape to (C x H x W)
            self.transforms = transforms.Compose([transforms.ToTensor()])
            data = [self.transforms(d) for d in data]
            data = torch.stack(data)
            with torch.no_grad():
                originals = data.to(self.device)
                if self.use_normals:
                    n = self._get_normals(originals.shape[0])
                    recs, _, _ = self.model(originals, n)
                else:
                    recs, _, _ = self.model(originals)
            recs = recs.permute(0,2,3,1)
            return recs.cpu().numpy()

        def display_images(image_paths, reference_img):
            import textwrap
            """Display the reference image and the most similar images."""
            # reference_image = imread(reference_image_path)

            images = [reference_img] + [imread(path) for path in image_paths]
            titles = ["Reference"] + [textwrap.fill((i.split('/')[-1]).split('.')[0],30) for i in image_paths]
            
            recs = reconstructions(np.array(images[1:]))
            # Create a figure with 2 rows and 5 columns
            fig, axs = plt.subplots(2, len(images), figsize=(15, 6))
            for ax in axs.flatten():
                ax.axis('off')

            axs[1, 0].imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
            axs[1, 0].set_title(titles[0])
            for i in range(1, len(images)):
                axs[0, i].imshow(images[i])
                axs[0, i].set_title(titles[i], wrap=True)
                axs[1, i].imshow(recs[i-1])
                axs[1, i].set_title('Reconstructed', wrap=True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, "similarity.png"))

        # Provide the path to the reference image and the dataset folder
        prior_samples = torch.rand(1, self.latent_dim) * 2 - 1 # range [-1,1]
        print(f'Prior samples used as reference: {prior_samples}')
        reference_img = self._decode_latents(prior_samples)
        reference_img = reference_img.permute(0,2,3,1)
        reference_img = reference_img.cpu().detach().numpy()
        reference_img = np.squeeze(reference_img)
        reference_img *= 255
        reference_img = reference_img.astype(np.uint8)
        plt.imshow(cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB))
        plt.waitforbuttonpress()
        print(f'Do you want to compute similarity on this sample? [Y/N]')
        answer = input().lower()
        while answer != 'y' and answer != 'n':
            print(f'Do you want to compute similarity on this sample? [Y/N]')
            answer = input().lower()
        if answer == "n":
            exit()
        print(f'Roger! Computing similarities... (this can take a few minutes)')
        dataset_folder = 'data/serrano/full-masked-serrano/'

        # Find the most similar images
        most_similar_images = find_most_similar_images(reference_img, dataset_folder)

        # Display the reference image and the most similar images
        display_images(most_similar_images, reference_img)


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
        
        gif_source = os.path.join(self.model_dir, "gif_source")
        create_safe_directory(gif_source)
        for image in range(len(all_cols)):
            filename = os.path.join(gif_source, f'frame_{image}.jpeg')
            imageio.imwrite(filename, all_cols[image])

        filename = os.path.join(self.model_dir, PLOT_NAMES["gif_traversals"])
        imageio.mimsave(filename, all_cols, fps=FPS_GIF)


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
        self.visualizer = Visualizer(model=model,
                                        model_dir=model_dir,
                                        dataset=dataset,
                                        use_normals=True,
                                        save_images=False,
                                        normal_img='ghost', **kwargs)

        self.images = []
        self.is_reorder_latents = is_reorder_latents
        self.n_per_latent = n_per_latent
        self.n_latents = n_latents if n_latents is not None else model.latent_dim

        self.freq = store_freq
        os.mkdir(f'{self.save_filename[:-12]}/traversals')
        self.n = 0 # Counter of the next isolated traversal image to be stored

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
        print(f'GifTraversalsTraining:save_reset() --> Not saving the gif, since it is so heavy (we should already have the individual traversals in a folder)')
        return 0
