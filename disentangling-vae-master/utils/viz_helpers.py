import random

import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import torch
import imageio

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils.datasets import get_dataloaders
from utils.helpers import set_seed

from utils.helpers import get_config_section
CONFIG_FILE = "hyperparam.ini"
configuration = get_config_section([CONFIG_FILE], "Testing")


FPS_GIF = 12


def get_samples(dataset, num_samples, idcs=[], random_ids=True, train_fld="Undefined", is_LAB=False, getting_vis=False):
    """ Generate a number of samples from the dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset.

    num_samples : int, optional
        The number of samples to load from the dataset

    idcs : list of ints, optional
        List of indices to of images to put at the begning of the samples.
    """

    print(f'Getting samples from {dataset} : {train_fld}')

    data_loader = get_dataloaders(dataset,
                                  batch_size=1,
                                  shuffle=idcs is None,
                                  train_fld=train_fld,
                                  is_LAB=is_LAB, 
                                  getting_vis=getting_vis)
    if random_ids:
        idcs += random.sample(range(len(data_loader.dataset)), num_samples - len(idcs))
    else:
        idcs = configuration['idcs']
    samples = torch.stack([data_loader.dataset[i][0] for i in idcs], dim=0)
    print("Selected idcs: {}".format(idcs))

    return samples

def get_dataset(dataset, name, results_path):
    """ Generate a number of samples from the dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    """

    print(f'Loading the dataset {name}...')
    data_loader = get_dataloaders(dataset,
                                  batch_size=1,
                                  shuffle=False)
    num_samples = len(data_loader)
    samples = torch.stack([data_loader.dataset[i][0] for i in range(num_samples)], dim=0)

    #? Reordered so that the channels are at the end (to respect the order in disentanglement_lib)
    samples_store = np.swapaxes(np.swapaxes(samples.detach().numpy(),1,2),2,3)

    data_path = f'{results_path}/dataset'
    print(f'Storing the dataset with shape {samples_store.shape} to {data_path}')
    np.save(data_path, samples_store)
        
    return samples.numpy()



def sort_list_by_other(to_sort, other, reverse=True):
    """Sort a list by an other."""
    return [el for _, el in sorted(zip(other, to_sort), reverse=reverse)]


# TO-DO: clean
def read_loss_from_file(log_file_path, loss_to_fetch):
    """ Read the average KL per latent dimension at the final stage of training from the log file.
        Parameters
        ----------
        log_file_path : str
            Full path and file name for the log file. For example 'experiments/custom/losses.log'.

        loss_to_fetch : str
            The loss type to search for in the log file and return. This must be in the exact form as stored.
    """
    EPOCH = "Epoch"
    LOSS = "Loss"

    logs = pd.read_csv(log_file_path)

    df_last_epoch_loss = logs[logs.loc[:, EPOCH] == logs.loc[:, EPOCH].max()]
    df_last_epoch_loss = df_last_epoch_loss.loc[df_last_epoch_loss.loc[:, LOSS].str.startswith(loss_to_fetch), :]
    # Modification for solving a _future_ warning
    # https://pandas.pydata.org/docs/dev/whatsnew/v1.5.0.html#inplace-operation-when-setting-values-with-loc-and-iloc
    df_last_epoch_loss.loc[:, LOSS] = df_last_epoch_loss.loc[:, LOSS].str.replace(loss_to_fetch, "").astype(int)


    df_last_epoch_loss = df_last_epoch_loss.sort_values(LOSS).loc[:, "Value"]
    return list(df_last_epoch_loss)


def add_labels(input_image, labels):
    """Adds labels next to rows of an image.

    Parameters
    ----------
    input_image : image
        The image to which to add the labels
    labels : list
        The list of labels to plot
    """
    new_width = input_image.width + 100
    new_size = (new_width, input_image.height)
    new_img = Image.new("RGB", new_size, color='white')
    new_img.paste(input_image, (0, 0))
    draw = ImageDraw.Draw(new_img)

    for i, s in enumerate(labels):
        draw.text(xy=(new_width - 100 + 0.005,
                      int((i / len(labels) + 1 / (2 * len(labels))) * input_image.height)),
                  text=s,
                  fill=(0, 0, 0))

    return new_img


def make_grid_img(tensor, **kwargs):
    """Converts a tensor to a grid of images that can be read by imageio.

    Notes
    -----
    * from in https://github.com/pytorch/vision/blob/master/torchvision/utils.py

    Parameters
    ----------
    tensor (torch.Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
        or a list of images all of the same size.

    kwargs:
        Additional arguments to `make_grid_img`.
    """
    grid = make_grid(tensor, **kwargs)
    img_grid = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    img_grid = img_grid.to('cpu', torch.uint8).numpy()
    return img_grid


def get_image_list(image_file_name_list):
    image_list = []
    for file_name in image_file_name_list:
        image_list.append(Image.open(file_name))
    return image_list


def arr_im_convert(arr, convert="RGBA"):
    """Convert an image array."""
    return np.asarray(Image.fromarray(arr).convert(convert))


def plot_grid_gifs(filename, grid_files, pad_size=7, pad_values=255):
    """Take a grid of gif files and merge them in order with padding."""
    grid_gifs = [[imageio.mimread(f) for f in row] for row in grid_files]
    n_per_gif = len(grid_gifs[0][0])

    # convert all to RGBA which is the most general => can merge any image
    imgs = [concatenate_pad([concatenate_pad([arr_im_convert(gif[i], convert="RGBA")
                                              for gif in row], pad_size, pad_values, axis=1)
                             for row in grid_gifs], pad_size, pad_values, axis=0)
            for i in range(n_per_gif)]

    imageio.mimsave(filename, imgs, fps=FPS_GIF)


def concatenate_pad(arrays, pad_size, pad_values, axis=0):
    """Concatenate lsit of array with padding inbetween."""
    pad = np.ones_like(arrays[0]).take(indices=range(pad_size), axis=axis) * pad_values

    new_arrays = [pad]
    for arr in arrays:
        new_arrays += [arr, pad]
    new_arrays += [pad]
    return np.concatenate(new_arrays, axis=axis)


class GaussianPlot:
# https://stackoverflow.com/questions/53152190/drawing-multiple-univariate-normal-distribution
  def __init__(self, title="Gaussian Distribution", x_label="x", y_label=None,
    y_limit=None, lower_bound=None, upper_bound=None, 
    with_grid=True, fill_below=True, legend_location="best"):
    
    self.title           = title
    self.x_label         = x_label
    self.y_label         = y_label or "N({}|μ,σ)".format(x_label)
    self.y_limit         = y_limit
    self.lower_bound     = lower_bound
    self.upper_bound     = upper_bound
    self.with_grid       = with_grid
    self.fill_below      = fill_below
    self.legend_location = legend_location
    
    self.plots = []

    # Color palette generated with https://mokole.com/palette.html
    self.colors = ['#696969','#2e8b57','#8b0000','#808000','#00008b','#ff0000','#ff8c00','#7fff00','#ba55d3','#00fa9a'
                    ,'#e9967a','#00ffff','#0000ff','#ff00ff','#1e90ff','#eee8aa','#ffff54','#dda0dd','#ff1493','#ff1493'
                    ,'#000000']  # ! Los colores que usamos para el plot
    

  def plot(self, mean, std, resolution=None, legend_label=None, line_width=1.0):
    self.plots.append({
      "mean":         mean,
      "std":          std,
      "resolution":   resolution,
      "legend_label": legend_label,
      "line_width": line_width
    })
    
    return self
  
  def show(self, filename="Undefined"):
    self._prepare_figure()
    self._draw_plots()
    
    plt.legend(loc=self.legend_location)
    plt.savefig(filename)
    print(f'Saved gaussians plot in {filename}')
  
  def _prepare_figure(self):
    plt.figure()
    plt.title(self.title)
    
    plt.xlabel(self.x_label)
    plt.ylabel(self.y_label)
    
    if self.y_limit is not None:
      plt.ylim(0, self.y_limit)
    
    if self.with_grid: 
      plt.grid()
  
  def _draw_plots(self):
    lower_bound = self.lower_bound if self.lower_bound is not None else self._compute_lower_bound()
    upper_bound = self.upper_bound if self.upper_bound is not None else self._compute_upper_bound()
    self.plot(0, 1, legend_label="N", resolution=5000, line_width=5.0) # *Add a last plot with the normal distribution
    counter = 0
    for plot_data in self.plots:
      mean         = plot_data["mean"]
      std          = plot_data["std"]
      resolution   = plot_data["resolution"]
      legend_label = plot_data["legend_label"]
      line_width   = plot_data["line_width"]
      
      self._draw_plot(lower_bound, upper_bound, mean, std, resolution, legend_label, counter, line_width)
      counter += 1
  
  def _draw_plot(self, lower_bound, upper_bound, mean, std, resolution, legend_label, c=0, width=1.0):
    resolution   = resolution or max(100, int(upper_bound - lower_bound)*10)
    legend_label = legend_label or "μ={}, σ={}".format(mean, std)
    
    X = np.linspace(lower_bound, upper_bound, resolution)
    dist_X = self._distribution(X, mean, std)
    
    if self.fill_below: plt.fill_between(X, dist_X, alpha=0.1)
    plt.plot(X, dist_X, label=legend_label, color=self.colors[c], linewidth=width)
  
  def _compute_lower_bound(self):
    return np.min([plot["mean"] - 4*plot["std"] for plot in self.plots])
  
  def _compute_upper_bound(self):
    return np.max([plot["mean"] + 4*plot["std"] for plot in self.plots])
  
  def _distribution(self, X, mean, std):
    return 1./(np.sqrt(2.*np.pi)*std)*np.exp(-np.power((X - mean)/std, 2.)/2)



def plot_gaussians(means, variances, sample_name="Undefined", filename="Undefined"):
    print(f'Plotting the distributions of all the {means.shape[1]} dimensions')
    variances = abs(variances)

    gaussian_plot = GaussianPlot(title=f"Latent Space Distributions", lower_bound=-2, upper_bound=2)
    for d in range(means.shape[1]):
        gaussian_plot.plot(means[0,d], variances[0,d], legend_label=d+1, resolution=5000)

    gaussian_plot.show(filename)
