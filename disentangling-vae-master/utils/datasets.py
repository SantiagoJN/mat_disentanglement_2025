import subprocess
import os
import abc
import hashlib
import zipfile
import glob
import logging
import tarfile
from skimage.io import imread
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import json

from utils.helpers import get_config_section, create_safe_directory

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from torch.nn.functional import conv2d
from torch.distributions import Normal
import math

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS_DICT = {"mnist": "MNIST",
                 "fashion": "FashionMNIST",
                 "dsprites": "DSprites",
                 "celeba": "CelebA",
                 "chairs": "Chairs",
                 "serrano": "Serrano",
                 "serranonormals": "SerranoNormals",
                 "datasetwnormals": "SerranoNormals",
                 "serranonormalsdisk": "SerranoNormalsDisk",
                 "serranotest": "SerranoTest",
                 "analytic": "Analytic",
                 "analyticnormals": "AnalyticNormals",
                 "guerrero": "Guerrero",
                 "wild": "Wild",
                 "custom": "Custom"}
DATASETS = list(DATASETS_DICT.keys())
CONFIG_FILE = "hyperparam.ini"

# train_folder = "Undefined training folder"

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

def gaussian_kernel(size: int, sigma: float):
    """Generate a 2D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1, size // 2 + 1)
    y = x[:, None]
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def apply_gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float):
    """Apply Gaussian blur to a color image."""
    # Ensure the image has a batch dimension
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dim

    # Generate the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.expand(image.size(1), 1, kernel_size, kernel_size)  # Depthwise convolution

    # Apply convolution with the Gaussian kernel
    blurred_image = F.conv2d(image.double(), kernel.double(), padding=kernel_size // 2, groups=image.size(1))

    return blurred_image.squeeze(0)  # Remove batch dim for output


def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).img_size


def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color


def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True, train_fld="Undefined", is_LAB=False, getting_vis=False, 
                    batch_size=128, triplets_name="Undefined", ratio=0, drop=0.0, logger=logging.getLogger(__name__), **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset : {"mnist", "fashion", "dsprites", "celeba", "chairs", "serrano", "analytic"}
        Name of the dataset to load

    root : str
        Path to the dataset root. If `None` uses the default one.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    global train_folder
    train_folder = train_fld

    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)
    dataset = Dataset(logger=logger, is_LAB=is_LAB, triplets_name=triplets_name, ratio=ratio, drop=drop) if root is None else Dataset(root=root, logger=logger, is_LAB=is_LAB, triplets_name=triplets_name, getting_vis=getting_vis)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **kwargs)


class DisentangledDataset(Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger


    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass


class DSprites(DisentangledDataset):
    """DSprites Dataset from [1].

    Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6
    disentangled latent factors. This dataset uses 6 latents, controlling the color,
    shape, scale, rotation and position of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.

    Notes
    -----
    - Link : https://github.com/deepmind/dsprites-dataset/
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
        M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
        with a constrained variational framework. In International Conference
        on Learning Representations.

    """
    urls = {"train": "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    files = {"train": "dsprite_train.npz"}
    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    background_color = COLOUR_BLACK
    lat_values = {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                  'orientation': np.array([0., 0.16110732, 0.32221463, 0.48332195,
                                           0.64442926, 0.80553658, 0.96664389, 1.12775121,
                                           1.28885852, 1.44996584, 1.61107316, 1.77218047,
                                           1.93328779, 2.0943951, 2.25550242, 2.41660973,
                                           2.57771705, 2.73882436, 2.89993168, 3.061039,
                                           3.22214631, 3.38325363, 3.54436094, 3.70546826,
                                           3.86657557, 4.02768289, 4.1887902, 4.34989752,
                                           4.51100484, 4.67211215, 4.83321947, 4.99432678,
                                           5.1554341, 5.31654141, 5.47764873, 5.63875604,
                                           5.79986336, 5.96097068, 6.12207799, 6.28318531]),
                  'shape': np.array([1., 2., 3.]),
                  'color': np.array([1.])}

    def __init__(self, root=os.path.join(DIR, '../data/dsprites/'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['imgs']
        self.lat_values = dataset_zip['latents_values']

    def download(self):
        """Download the dataset."""
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", self.train_data])

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample = self.transforms(sample)

        lat_value = self.lat_values[idx]
        return sample, lat_value


class CelebA(DisentangledDataset):
    """CelebA Dataset from [1].

    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.
    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including
    10,177 number of identities, and 202,599 number of face images.

    Notes
    -----
    - Link : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face
        attributes in the wild. In Proceedings of the IEEE international conference
        on computer vision (pp. 3730-3738).

    """
    urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    files = {"train": "img_align_celeba"}
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/celeba'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + '/*')

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'celeba.zip')
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", save_path])

        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
            '{} file is corrupted.  Remove the file and try again.'.format(save_path)

        with zipfile.ZipFile(save_path) as zf:
            self.logger.info("Extracting CelebA ...")
            zf.extractall(self.root)

        os.remove(save_path)

        self.logger.info("Resizing CelebA ...")
        preprocess(self.train_data, size=type(self).img_size[1:])

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0


class Serrano(DisentangledDataset):
    """Serrano Dataset from [1].

    Serrano et al. dataset is composed by a set of renderings using MERL dataset 
    along with other materials obtained out of it, using a number of 3D models
    and environmental illumination maps.

    Notes
    -----
    - Link : https://mig.mpi-inf.mpg.de/

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Ana Serrano, Bin Chen, Chao Wang, Michal Piovarči, Hans-Peter Seidel, 
    Piotr Didyk, and Karol Myszkowski. 2021. The effect of shape and 
    illumination on material perception: model and applications. 
    ACM Trans. Graph. 40, 4, Article 125 (August 2021), 16 pages. 

    """
    default_config = get_config_section([CONFIG_FILE], "Datasets")
    #urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    img_size = default_config['img_size'] 
    background_color = COLOUR_WHITE

    use_labels = default_config['use_labels'] # To switch using labels or not 
    labels = []
    def __init__(self, root=os.path.join(DIR, '../data/serrano'), is_LAB=None, triplets_name=None, ratio=0, drop=0, **kwargs):
        #print(f"INFO: Initiating with the directory {root}")
        if is_LAB != None:
            print("[WARNING] Calling the vanilla Serrano dataset with is_LAB; If it's not for inference, replace -d with \'serranonormals\'")
        self.is_LAB = is_LAB
        Serrano.files = {"train": train_folder} # Programado para que se pueda configurar por los parámetros de entrada
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        from natsort import os_sorted # windows explorer-like sorting..
        self.imgs = os_sorted(glob.glob(self.train_data + '/*'))

        if self.use_labels:
            print(f'INFO: Using geometry labels')
            self.labels = [5 if 'blob' in sample else 1 for sample in self.imgs]
            
        else:
            self.labels = np.zeros(len(self.imgs)) # Always give 0s
        

    def download(self):
        print("###############################################################")
        print("##### WARNING: TRYING TO DOWNLOAD SERRANO ET AL. DATASET! #####")
        print("###############################################################")

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """
        
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        if self.is_LAB:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB) # convert to Lab

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        return img, img, img # No normal map, and objective=img


class SerranoNormals(DisentangledDataset):
    """ Modification of the Serrano Dataset to use normal maps
    
    Serrano Dataset from [1].

    Serrano et al. dataset is composed by a set of renderings using MERL dataset 
    along with other materials obtained out of it, using a number of 3D models
    and environmental illumination maps.

    Notes
    -----
    - Link : https://mig.mpi-inf.mpg.de/

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Ana Serrano, Bin Chen, Chao Wang, Michal Piovarči, Hans-Peter Seidel, 
    Piotr Didyk, and Karol Myszkowski. 2021. The effect of shape and 
    illumination on material perception: model and applications. 
    ACM Trans. Graph. 40, 4, Article 125 (August 2021), 16 pages. 

    """
    default_config = get_config_section([CONFIG_FILE], "Datasets")
    #urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    img_size = default_config['img_size'] 
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/serrano'), is_LAB=False, triplets_name=None, ratio=0, drop=0, **kwargs):
        SerranoNormals.files = {"train": train_folder} # Programado para que se pueda configurar por los parámetros de entrada

        super().__init__(root, [transforms.ToTensor()], **kwargs)
        precomputed_triplets = f"data/{triplets_name}.json" # Get the precomputed triplets that we'll use to train the model
        
        
        with open(precomputed_triplets) as f: 
            print(f"Reading triplets in {precomputed_triplets} ...")
            self.data_triplets = json.load(f)
        
        self.num_triplets = len(self.data_triplets["sample"])
        self.num_samples = self.data_triplets["num_samples"]
        self.num_goal_geo = self.data_triplets["num_goal_geometries"]
        print(f"[INFO] {self.num_triplets} triplets successfully loaded! ({self.num_samples} x {self.num_goal_geo})")
        self.imgs = [name_sample for i, name_sample in enumerate(self.data_triplets["sample"]) if i%self.num_goal_geo == 0]
        self.imgs.sort()
        
        self.normals_loaded = []
        self.imgs_loaded = []
        print("loading all the images!")
        if is_LAB:
            print("    ...and converting them to CIE Lab color space :)")

        # * LOADING NORMALS
        for normal_path in self.data_triplets["normals_paths"]:
            self.normals_loaded.append(self.transforms((imread(normal_path))[:,:,:3]))
            
        # * LOADING IMAGES
        total_samples = len(self.data_triplets["samples_paths"])
        step = int(total_samples / 10)
        for idx, im_path in enumerate(self.data_triplets["samples_paths"]): # Load all the individual images in the dataset
            if idx % step == 0:
                print(f'[{idx} / {total_samples}]')
            image = cv2.imread(im_path)
            if is_LAB:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB) # convert to Lab
            image = self.transforms(image) # apply transforms
            self.imgs_loaded.append(image) # and store it for later :)

        # * COMPUTING FILTERS (if that's the case)
        self.drop_filters = drop # once every N times, we will drop some pixels
        if self.drop_filters != 0:
            self.num_filters = 64
            self.filters = []
            total_pixels = self.normals_loaded[0].shape[1] * self.normals_loaded[0].shape[2]
            pixels_to_remove = int(ratio * total_pixels)
            for f in range(self.num_filters):
                filt = torch.from_numpy(np.ones(self.normals_loaded[0].shape)) # [1:] to ignore the batch size
                rm_coords = np.random.choice(self.normals_loaded[0].shape[1], size=pixels_to_remove*2)
                for i in range(pixels_to_remove):
                    filt[:,rm_coords[i*2],rm_coords[i*2+1]] = 0
                filt = gaussian_blur(filt, 5) # Set clusters of black pixels, instead of sparse ones.
                filt[filt < 1] = 0
                self.filters.append(filt)

        print(f"[INFO] Ended loading {len(self.imgs_loaded)} images!")
        

    def download(self):
        print("###############################################################")
        print("##### WARNING: TRYING TO DOWNLOAD SERRANONORMALS DATASET! #####")
        print("###############################################################")

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """

        # we can imagine our triplet dataframe as a table with size nxm where n are the number
        # of samples, and m is the number of geometries. here, idx represents the index in n
        selected_geometry = random.randint(0,self.num_goal_geo-1) # random number in range [0..num_goal_geo]
        selected_triplet = idx * (self.num_goal_geo-1) + selected_geometry # manually building the index to access this table
        if selected_triplet >= self.num_triplets or selected_triplet > len(self.data_triplets["sample"]):
            print(f'Tried to access an invalid triplet: {selected_triplet}. idx: {idx}, num_goal_geo = {self.num_goal_geo-1}, selected_geometry: {selected_geometry}')
            print(f'num_triplets: {self.num_triplets}, and len of data_triplets[sample] is {len(self.data_triplets["sample"])}')

        # Get the sample
        img = self.imgs_loaded[self.data_triplets["sample"][selected_triplet]]

        # Get the normal
        normal = self.normals_loaded[self.data_triplets["normal"][selected_triplet]]

        # Get the objective
        objective = self.imgs_loaded[self.data_triplets["objective"][selected_triplet]]

        
        if self.drop_filters != 0 and idx % self.drop_filters == 0: # 1/N of the times, remove a random number of pixels
            selected_filter = self.filters[random.randint(0,self.num_filters-1)]
            normal *= selected_filter
            objective *= selected_filter

        """ #COOL WAY TO VISUALIZE WHAT GOES INTO THE NETWORK
        cv2.imshow("img",img.permute(1,2,0).numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imshow("normal",normal.permute(1,2,0).numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imshow("objective",objective.permute(1,2,0).numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        return img, normal, objective


class SerranoNormalsDisk(DisentangledDataset):
    """ 
        Modification made to the SerranoNormals dataset, with the objective of 
        not storing every training sample in memory, but read them along the 
        training, like it is done in the default dataset.
        With this, we can run the training with big (1024x1024) samples, which
        otherwise would occupy ~250GB of ram.

    """
    default_config = get_config_section([CONFIG_FILE], "Datasets")
    #urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    img_size = default_config['img_size'] 
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/serrano'), is_LAB=False, triplets_name=None, **kwargs):
        SerranoNormalsDisk.files = {"train": train_folder} # Programado para que se pueda configurar por los parámetros de entrada
        
        super().__init__(root, [transforms.ToTensor()], **kwargs)
        precomputed_triplets = f"data/serrano/{triplets_name}.json"
        
        with open(precomputed_triplets) as f: 
            print("Reading triplets json...")
            self.data_triplets = json.load(f)
        
        self.num_triplets = len(self.data_triplets["sample"])
        self.num_samples = self.data_triplets["num_samples"]
        self.num_goal_geo = self.data_triplets["num_goal_geometries"]
        print(f"[INFO] {self.num_triplets} triplets successfully loaded! ({self.num_samples} x {self.num_goal_geo})")
        self.imgs = [name_sample for i, name_sample in enumerate(self.data_triplets["sample"]) if i%self.num_goal_geo == 0]
        self.imgs.sort()

        
        

    def download(self):
        print("###############################################################")
        print("##### WARNING: TRYING TO DOWNLOAD SERRANONORMALSDISK DATASET! #####")
        print("###############################################################")

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """

        selected_geometry = random.randint(0,self.num_goal_geo-1) # random number in range [0..num_goal_geo]
        selected_triplet = idx * self.num_goal_geo-1 + selected_geometry # manually building the index to access this table

        # Get the sample
        img_path = self.data_triplets["samples_paths"][self.data_triplets["sample"][selected_triplet]]
        img = imread(img_path) # * Input X (illumination + material)
        img = self.transforms(img)
        
        # Get the normal
        normal_path = self.data_triplets["normals_paths"][self.data_triplets["normal"][selected_triplet]]
        normal = imread(normal_path) # Read the normal map
        normal = normal[:,:,:3] # Remove the alpha channel from the png image
        normal = self.transforms(normal) # Transform it to make it 

        # Get the objective
        objective_path = self.data_triplets["samples_paths"][self.data_triplets["objective"][selected_triplet]]
        objective = imread(objective_path)
        objective = self.transforms(objective)

        return img, normal, objective



class AnalyticNormals(DisentangledDataset):

    default_config = get_config_section([CONFIG_FILE], "Datasets")
    #urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    img_size = default_config['img_size'] 
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/analytic'), is_LAB=False, getting_vis=False, **kwargs):
        AnalyticNormals.files = {"train": train_folder} # Programado para que se pueda configurar por los parámetros de entrada

        super().__init__(root, [transforms.ToTensor()], **kwargs)
        
        self.getting_vis = getting_vis
        precomputed_triplets = "data/analytic/precomputed_triplets_analytic_local.json"
        with open(precomputed_triplets) as f: 
            print("Reading triplets json...")
            self.data_triplets = json.load(f)
        
        self.num_triplets = len(self.data_triplets["sample"])
        self.num_samples = self.data_triplets["num_samples"]
        self.num_goal_geo = self.data_triplets["num_goal_geometries"]

        print(f'Loading images...')
        self.imgs = []
        root_path_imgs = f'{root}/{self.data_triplets["path_images"]}'
        for path in os.listdir(root_path_imgs):
            image = cv2.imread(f'{root_path_imgs}/{path}')
            image = self.transforms(image) # apply transforms
            self.imgs.append(image) # and store it for later :)

        print(f'and normals...')
        self.normals = []
        root_path_imgs = f'{root}/{self.data_triplets["path_normals"]}'
        for path in os.listdir(root_path_imgs):
            image = cv2.imread(f'{root_path_imgs}/{path}')
            image = self.transforms(image) # apply transforms
            self.normals.append(image) # and store it for later :)
            
        print(f'    ....done!')
        
        self.images_idcs = self.data_triplets["sample"]
        self.normals_idcs = self.data_triplets["normal"]
        self.goals_idcs = self.data_triplets["objective"]
        
        print(f"[INFO] {self.num_triplets} triplets successfully loaded! ({self.num_samples} x {self.num_goal_geo})")




    def download(self):
        print("###############################################################")
        print("##### WARNING: TRYING TO DOWNLOAD ANALYTIC DATASET! #####")
        print("###############################################################")

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """

        # we can imagine our triplet dataframe as a table with size nxm where n are the number
        # of samples, and m is the number of geometries. here, idx represents the index in n
        if self.getting_vis:
            selected_geometry = idx % self.num_goal_geo-1 # deterministic sample
        else:
            selected_geometry = random.randint(0,self.num_goal_geo-1) # random number in range [0..num_goal_geo]
        selected_triplet = idx * self.num_goal_geo-1 + selected_geometry # manually building the index to access this table
        
        # Get the sample
        img = self.imgs[self.data_triplets["sample"][selected_triplet]]

        # Get the normal
        normal = self.normals[self.data_triplets["normal"][selected_triplet]]

        # Get the objective
        objective = self.imgs[self.data_triplets["objective"][selected_triplet]]

        return img, normal, objective



class Guerrero(DisentangledDataset):
    
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '/media/raid/dsubias/latent-data'), **kwargs):
        default_config = get_config_section([CONFIG_FILE], "Datasets")
        img_size = default_config['img_size'] 
        
        Guerrero.files = {"train": train_folder} # Programado para que se pueda configurar por los parámetros de entrada
        super().__init__(root, [transforms.ToPILImage(),transforms.Resize(img_size[2]), transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + '/*')
        self.imgs.sort() # Sort it so the order of the images is predictable
        
        self.labels = np.zeros(len(self.imgs)) # Always give 0s
        

    def download(self):
        print("###############################################################")
        print("##### WARNING: TRYING TO DOWNLOAD GUERRERO ET AL. DATASET! #####")
        print("###############################################################")

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        normal = imread("data/serrano/geometry-normals/teapot.png")
        normal = normal[:,:,:3] # Remove the alpha channel from the png image
        normal = self.transforms(normal)

        return img, normal, img # TODO: Remove this unnecessary third img?

class SerranoTest(DisentangledDataset):
    """
    A little helper to load the test dataset composed by some selected images from
    Serrano et al. More information in Serrano dataset comment.
    """
    files = {"train": "test"}
    #                     ^---HERE GOES THE FOLDER WITH TEST IMAGES
    
    img_size = (3, 256, 256)
    background_color = COLOUR_WHITE

    use_labels = False # To switch using labels or not 
    labels = []
    def __init__(self, root=os.path.join(DIR, '../data/serrano'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + '/*')

        if self.use_labels:
            print(f'WARNING: Using labels')
            labels_path = os.path.join(root, "glossiness_labels.npy")
            self.labels = np.load(labels_path)
        else:
            self.labels = np.zeros(len(self.imgs)) # Always give 0s
        

    def download(self):
        print("###############################################################")
        print("##### WARNING: TRYING TO DOWNLOAD SERRANO ET AL. DATASET! #####")
        print("###############################################################")

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        # no label so return 0 (note that can't return None because) dataloaders requires so
        return img, self.labels[idx]  # It will return label = 0 if it doesn't use labels


class Analytic(DisentangledDataset):
    """Analytic dataset Dataset from [1].

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Predicting Perceived Gloss: Do Weak Labels Suffice? Guerrero-Viu, 
    Subias, Serrano, Storrs, Fleming, Masia, Gutierrez. Under submission 

    """
    default_config = get_config_section([CONFIG_FILE], "Datasets")
    files = {"train": default_config['training_dataset']}
    img_size = default_config['img_size'] 
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/analytic'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + '/*')
        self.imgs.sort() # ? Sort it so the order of the images is predictable

        self.labels = np.zeros(len(self.imgs)) # Always give 0s
        

    def download(self):
        print("###############################################################")
        print("######## WARNING: TRYING TO DOWNLOAD ANALYTIC DATASET! ########")
        print("###############################################################")

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        return img, self.labels[idx]  # It will return label = 0 if it doesn't use labels


class Wild(DisentangledDataset):
    """Wild dataset.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References

    """
    default_config = get_config_section([CONFIG_FILE], "Datasets")
    files = {"train": default_config['training_dataset']}
    img_size = default_config['img_size'] 
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/wild'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        print(self.train_data)
        self.imgs = glob.glob(self.train_data + '/*')
        self.imgs.sort() # ? Sort it so the order of the images is predictable

        self.labels = np.zeros(len(self.imgs)) # Always give 0s
        

    def download(self):
        print("###############################################################")
        print("######## WARNING: TRYING TO DOWNLOAD WILD DATASET! ########")
        print("###############################################################")

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        return img, self.labels[idx]  # It will return label = 0 if it doesn't use labels


class Custom(DisentangledDataset):
    """Custom dataset, made from mixing in some way the Serrano and Analytic datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References

    """
    default_config = get_config_section([CONFIG_FILE], "Datasets")
    files = {"train": default_config['training_dataset']}
    img_size = default_config['img_size'] 
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/custom'), is_LAB=False, **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + '/*')
        self.imgs.sort() # ? Sort it so the order of the images is predictable

        self.labels = np.zeros(len(self.imgs)) # Always give 0s
        

    def download(self):
        print("###############################################################")
        print("######## WARNING: TRYING TO DOWNLOAD CUSTOM DATASET! ########")
        print("###############################################################")

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        return img, img, img  # It will return label = 0 if it doesn't use labels



class Chairs(datasets.ImageFolder):
    """Chairs Dataset from [1].

    Notes
    -----
    - Link : https://www.di.ens.fr/willow/research/seeing3Dchairs

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Aubry, M., Maturana, D., Efros, A. A., Russell, B. C., & Sivic, J. (2014).
        Seeing 3d chairs: exemplar part-based 2d-3d alignment using a large dataset
        of cad models. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 3762-3769).

    """
    urls = {"train": "https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar"}
    files = {"train": "chairs_64"}
    img_size = (1, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/chairs'),
                 logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose([transforms.Grayscale(),
                                              transforms.ToTensor()])
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

        super().__init__(self.train_data, transform=self.transforms)

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'chairs.tar')
        os.makedirs(self.root)
        subprocess.check_call(["curl", type(self).urls["train"],
                               "--output", save_path])

        self.logger.info("Extracting Chairs ...")
        tar = tarfile.open(save_path)
        tar.extractall(self.root)
        tar.close()
        os.rename(os.path.join(self.root, 'rendered_chairs'), self.train_data)

        os.remove(save_path)

        self.logger.info("Preprocessing Chairs ...")
        preprocess(os.path.join(self.train_data, '*/*'),  # root/*/*/*.png structure
                   size=type(self).img_size[1:],
                   center_crop=(400, 400))


class MNIST(datasets.MNIST):
    """Mnist wrapper. Docs: `datasets.MNIST.`"""
    img_size = (1, 32, 32)
    background_color = COLOUR_BLACK

    def __init__(self, root=os.path.join(DIR, '../data/mnist'), **kwargs):
        super().__init__(root,
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor()
                         ]))


class FashionMNIST(datasets.FashionMNIST):
    """Fashion Mnist wrapper. Docs: `datasets.FashionMNIST.`"""
    img_size = (1, 32, 32)
    background_color = COLOUR_BLACK

    def __init__(self, root=os.path.join(DIR, '../data/fashionMnist'), **kwargs):
        super().__init__(root,
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor()
                         ]))


# HELPERS
def preprocess(root, size=(64, 64), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)
