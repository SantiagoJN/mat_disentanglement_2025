# Based in https://deepnote.com/@econdesousa/PCA-using-custom-class-and-sklearn-0bb84f80-8051-4987-bed0-0319d80a4715
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Based in https://danielmuellerkomorowska.com/2021/01/05/introduction-to-t-sne-in-python-with-scikit-learn/
import argparse
import os
import sys
from PIL import Image
from matplotlib import pyplot as plt
from random import randint
from random import sample
import numpy as np
import torch

from utils.helpers import create_safe_directory
from utils.helpers import FormatterNoDuplicate, check_bounds, set_seed
from main import RES_DIR
from disvae.utils.modelIO import load_model, load_metadata
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"]="0"

PLOT_TYPES = ['generate-samples', 'data-samples', 'reconstruct', "traversals",
              'reconstruct-traverse', "gif-traversals", "all"]

device = torch.device('cuda:0') #get_device(is_gpu=not args.no_cuda)
print(f'----Using {torch.cuda.get_device_name(0)}----')

def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    description = "CLI for plotting using pretrained models of `disvae`"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    parser.add_argument('name', type=str,
                        help="Name of the model for storing and loading purposes.")
    parser.add_argument('-d', '--dataset', type=str, default='data/serrano/color_chnl',
                        help="Location of the dataset we want to get the TSNE plot from")
    args = parser.parse_args()

    return args

def main(args):
    """Main function for plotting fro pretrained models.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    
    #### HYPERPARAMETERS ####
    batch_size = 1024
    num_batches = 3
    get_means = True # Save the TSNE plots obtained only with the means
    get_samples = False # Save the TSNE plots obtained with samples
    custom_subset = True # Use a subset of the serrano dataset or use all


    experiment_name = args.name
    model_dir = os.path.join(RES_DIR, experiment_name)
    meta_data = load_metadata(model_dir)
    model = load_model(model_dir)
    print(f'Model in {model_dir}')
    exit
    model.eval()  # don't sample from latent: use mean
    print(f'Experiment name: {experiment_name}')
    #print(model.encoder)

    list_names = sorted(os.listdir(args.dataset))
    imgs = np.load("data/serrano/images_data.npy") # Here we always load the whole dataset of images
    
    print(f'Hay {len(imgs)} imagenes con un shape de {imgs[0].shape}')


    # For this, we will need a .npy containing the indices that we will want.
    if custom_subset: # ! Do this if we want to run TSNE in a specific subset of data.
        subset_name = "data/serrano/mini-serrano_idx.npy"
        indices = np.load(subset_name)
        print(f'Using the subset {subset_name}')
    else:
        indices = range(len(imgs)) # Get all them

    imgs = imgs[indices] # Filter the images we really want
    

    imgs = np.swapaxes(imgs,1,3) # Flip to have the #channels as first dimension
    selected_idcs = sample(range(0, len(imgs)), batch_size*num_batches)
    latent_mean = np.zeros((batch_size*num_batches,20), dtype=float) # Save space to store intermediate results
    latent_sample = np.zeros((batch_size*num_batches,20), dtype=float) # To store samples instead of just means
    for batch in range(num_batches): # Get the latent representation in batches
        print(f'Getting representations of batch No.{batch}')
        torch.cuda.empty_cache()
    # Check the shapes of the images and the labels
        batch_imgs = imgs[selected_idcs[batch*batch_size:(batch+1)*batch_size],:,:,:] # Get a batch of 64 images
        batch_imgs = torch.from_numpy(batch_imgs) # Convert it to a torch tensor
        batch_imgs = batch_imgs.to(device) # Send it to the device
        batch_imgs = batch_imgs.float() # Converting it into float because otherwise it outputs errors

        # Introduce every image to the encoder and check that the output is 20D
        with torch.no_grad():
            mean, variance = model.encoder(batch_imgs) # Finally get the latent representation

        mean = mean.to(torch.device("cpu")) # Send it back to the cpu
        mean = mean.detach().numpy()
        variance = variance.to(torch.device("cpu")) # Send it back to the cpu
        variance = variance.detach().numpy()
        
        if get_means:
            latent_mean[batch*batch_size:(batch+1)*batch_size, :] = mean # Save the intermediate result
    


    scaled_data = StandardScaler().fit_transform(latent_mean)
    #os.exit()
    print('Running PCA...')
    n_components = 20 # In PCA we want it to explain the variance of the 20
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(scaled_data)
    principalDf = pd.DataFrame(principalComponents, columns=['PC'+str(i+1) for i in range(n_components)])
    explained_variance = pca.explained_variance_ratio_ # !Variance ratio!
    print("explained_variance per principal component [%]: ", np.round(explained_variance*100,decimals=2))
    print("sum of explained_variance [%]: ", np.round(sum(explained_variance)*100, decimals=2))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.bar(
        x      = np.arange(n_components) + 1,
        height = pca.explained_variance_ratio_
    )

    for x, y in zip(np.arange(20) + 1, pca.explained_variance_ratio_):
        label = round(y, 2)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )

    ax.set_xticks(np.arange(n_components) + 1)
    ax.set_ylim(0, 1.1)
    ax.set_title('Variance explained per component')
    ax.set_xlabel('Component')
    ax.set_ylabel('%')

    plt.show()


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
