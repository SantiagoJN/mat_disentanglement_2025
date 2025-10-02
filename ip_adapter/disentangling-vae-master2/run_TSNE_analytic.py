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

def load_labels(root_path, indices=None):
    anisotropy_label = np.load(os.path.join(root_path, 'anisotropy_labels.npy'))
    contgloss_label = np.load(os.path.join(root_path, 'contgloss_labels.npy'))
    geometry_label = np.load(os.path.join(root_path, 'geometry_labels.npy'))
    glossiness_label = np.load(os.path.join(root_path, 'glossiness_labels.npy'))
    illumination_label = np.load(os.path.join(root_path, 'illumination_labels.npy'))
    lightness_label = np.load(os.path.join(root_path, 'lightness_labels.npy'))
    metallicness_label = np.load(os.path.join(root_path, 'metallicness_labels.npy'))
    refsharp_label = np.load(os.path.join(root_path, 'refsharp_labels.npy'))

    # Filter the labels we want
    anisotropy_label = anisotropy_label[indices]
    contgloss_label = contgloss_label[indices]
    geometry_label = geometry_label[indices]
    glossiness_label = glossiness_label[indices]
    illumination_label = illumination_label[indices]
    lightness_label = lightness_label[indices]
    metallicness_label = metallicness_label[indices]
    refsharp_label  = refsharp_label[indices]

    return anisotropy_label, contgloss_label, geometry_label, glossiness_label, \
            illumination_label, lightness_label, metallicness_label, refsharp_label

def replace_labels(label_array, old_val, new_val):
    label_array[label_array==old_val] = new_val
    return label_array

def plot_TSNE(tsne_result, labels, title, root):
    
    # Change the labels to make the legend more intuitive
    if "geometry" in title:
        labels = labels.astype('str')
        labels = replace_labels(labels, "0", "blob")
        labels = replace_labels(labels, "1", "sphere")
    elif "illumination" in title:
        labels = labels.astype('str')
        labels = replace_labels(labels, "0", "circus")
        labels = replace_labels(labels, "1", "art_studio")

    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': labels})
    fig, ax = plt.subplots(1)
    plt.title(title)
    # https://seaborn.pydata.org/generated/seaborn.scatterplot.html
    scatter = sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120, legend="full")
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig(os.path.join(root, title))
    # plt.show()
    # plt.waitforbuttonpress()

# From the vae.py file
def reparameterize(mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        """
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*std.shape) # Ensure that eps has the same shape as std
        return mean + std * eps.astype(np.float32)

def main(args):
    """Main function for plotting fro pretrained models.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    
    #### HYPERPARAMETERS ####
    batch_size = 600
    num_batches = 1
    get_means = True # Save the TSNE plots obtained only with the means
    get_samples = False # Save the TSNE plots obtained with samples


    experiment_name = args.name
    model_dir = os.path.join(RES_DIR, experiment_name)
    meta_data = load_metadata(model_dir)
    model = load_model(model_dir)
    print(f'Model in {model_dir}')
    # exit
    model.eval()  # don't sample from latent: use mean
    print(f'Experiment name: {experiment_name}')
    #print(model.encoder)

    list_names = sorted(os.listdir(args.dataset))
    imgs = np.load("/home/santiagojn/disentangling-vae-master/data/custom/trial3/test_analytic_packed.npy")
    print(f'Hay {len(imgs)} imagenes con un shape de {imgs[0].shape}')

    values = np.hstack([np.repeat(0,3),np.repeat(1,3),np.repeat(2,3),np.repeat(3,3),np.repeat(4,3),np.repeat(5,3),np.repeat(6,3),np.repeat(7,3),np.repeat(8,3),np.repeat(9,3)])
    specular_label = np.tile(values,20)
    values = np.hstack([np.repeat(0,30),np.repeat(1,30),np.repeat(2,30),np.repeat(3,30),np.repeat(4,30)])
    rough_label = np.tile(values,4)
    values = np.hstack([np.repeat(0,150),np.repeat(1,150)])
    geometry_label = np.tile(values,2)
    illumination_label = np.hstack([np.repeat(0,300),np.repeat(1,300)])

    print(specular_label)
    print("^specular^=========================")
    print(rough_label)
    print("^rough^=========================")
    print(geometry_label)
    print("^geometry^=========================")
    print(illumination_label)
    print("^illumination^=========================")
    

    imgs = np.swapaxes(imgs,1,3) # Flip to have the #channels as first dimension
    selected_idcs = list(range(600)) #sample(range(0, len(imgs)), batch_size*num_batches)
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
        #print(mean.shape)

        # Use TSNE to transform those 20D into 2D
        mean = mean.to(torch.device("cpu")) # Send it back to the cpu
        mean = mean.detach().numpy()
        variance = variance.to(torch.device("cpu")) # Send it back to the cpu
        variance = variance.detach().numpy()
        
        if get_means:
            latent_mean[batch*batch_size:(batch+1)*batch_size, :] = mean # Save the intermediate result
        if get_samples:
            latent_sample[batch*batch_size:(batch+1)*batch_size, :] = reparameterize(mean,variance)
        
        #? Each 10 batches, sample 10 times the current batch to get a representative representation
        # if batch % 10 == 0:
        #     for i in range(10): # Sample 10 times
        #         latent_sample[(batch)*batch_size+i:(batch+1)*batch_size+i, :] = reparameterize(mean,variance)

    #os.exit()
    print('Running TSNE...')
    n_components = 2 # This is always going to be the case
    tsne = TSNE(n_components) # Declare TSNE object
    if get_means:
        tsne_mean = tsne.fit_transform(latent_mean) # Apply it
    if get_samples:
        tsne_sample = tsne.fit_transform(latent_sample) # Also for the sampled one
    # print(f'tsne: {tsne_result.shape}')
    # print(f'Geomtery labels: {(geometry_label[selected_idcs]).shape}')



    # Show the different plots using every label
    TSNE_folder = os.path.join(os.path.join(RES_DIR, experiment_name), "TSNE")
    create_safe_directory(TSNE_folder)
    print(f'Saving plots...')
    if get_means:
        plot_TSNE(tsne_mean, geometry_label, "mean_2_geometry", TSNE_folder)
        plot_TSNE(tsne_mean, illumination_label, "mean_1_illumination", TSNE_folder)
        plot_TSNE(tsne_mean, rough_label, "mean_rough", TSNE_folder)
        plot_TSNE(tsne_mean, specular_label, "mean_specular", TSNE_folder)
        print("    Saved plots for the means")
    if get_samples:
        plot_TSNE(tsne_sample, geometry_label[selected_idcs], "sample_2_geometry", TSNE_folder)
        plot_TSNE(tsne_sample, illumination_label[selected_idcs], "sample_1_illumination", TSNE_folder)
        plot_TSNE(tsne_sample, glossiness_label[selected_idcs], "sample_3_glossiness", TSNE_folder)
        print("    Saved plots for the samples")
    
    
    # plot_TSNE(tsne_result, refsharp_label[selected_idcs], "Sharpness of reflections", TSNE_folder)
    # plot_TSNE(tsne_result, contgloss_label[selected_idcs], "Contrast of reflections", TSNE_folder)
    # plot_TSNE(tsne_result, metallicness_label[selected_idcs], "Metallicness", TSNE_folder)
    # plot_TSNE(tsne_result, lightness_label[selected_idcs], "Lightness", TSNE_folder)
    # plot_TSNE(tsne_result, anisotropy_label[selected_idcs], "Anisotropy", TSNE_folder)

    if get_means:
        np.savetxt(os.path.join(TSNE_folder, 'tsne_mean.txt'), tsne_mean)
    if get_samples:
        np.savetxt(os.path.join(TSNE_folder, 'tsne_sample.txt'), tsne_sample)

    print(f'Plots saved in {TSNE_folder}')

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
