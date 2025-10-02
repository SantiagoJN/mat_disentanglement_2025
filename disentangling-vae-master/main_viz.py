import argparse
import os
import sys

from utils.helpers import FormatterNoDuplicate, check_bounds, set_seed
from utils.visualize import Visualizer
from utils.viz_helpers import get_samples, get_dataset
from main import RES_DIR
from disvae.utils.modelIO import load_model, load_metadata

from utils.helpers import get_config_section
CONFIG_FILE = "hyperparam.ini"
configuration = get_config_section([CONFIG_FILE], "Testing")
dataset_config = get_config_section([CONFIG_FILE], "Datasets")

PLOT_TYPES = ['generate-samples', 'data-samples', 'reconstruct', "traversals",
              'reconstruct-traverse', "gif-traversals", "all"]


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
    parser.add_argument("plots", type=str, nargs='+', choices=PLOT_TYPES,
                        help="List of all plots to generate. `generate-samples`: random decoded samples. `data-samples` samples from the dataset. `reconstruct` first rnows//2 will be the original and rest will be the corresponding reconstructions. `traversals` traverses the most important rnows dimensions with ncols different samples from the prior or posterior. `reconstruct-traverse` first row for original, second are reconstructions, rest are traversals. `gif-traversals` grid of gifs where rows are latent dimensions, columns are examples, each gif shows posterior traversals. `all` runs every plot.")
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed. Can be `None` for stochastic behavior.')
    parser.add_argument('-r', '--n-rows', type=int, default=6,
                        help='The number of rows to visualize (if applicable).')
    parser.add_argument('-c', '--n-cols', type=int, default=7,
                        help='The number of columns to visualize (if applicable).')
    parser.add_argument('-t', '--max-traversal', default=2,
                        type=lambda v: check_bounds(v, lb=0, is_inclusive=False,
                                                    type=float, name="max-traversal"),
                        help='The maximum displacement induced by a latent traversal. Symmetrical traversals are assumed. If `m>=0.5` then uses absolute value traversal, if `m<0.5` uses a percentage of the distribution (quantile). E.g. for the prior the distribution is a standard normal so `m=0.45` corresponds to an absolute value of `1.645` because `2m=90%%` of a standard normal is between `-1.645` and `1.645`. Note in the case of the posterior, the distribution is not standard normal anymore.')
    parser.add_argument('-i', '--idcs', type=int, nargs='+', default=[],
                        help='List of indices to of images to put at the begining of the samples.')
    parser.add_argument('-u', '--upsample-factor', default=1,
                        type=lambda v: check_bounds(v, lb=1, is_inclusive=True,
                                                    type=int, name="upsample-factor"),
                        help='The scale factor with which to upsample the image (if applicable).')
    parser.add_argument('--is-show-loss', action='store_true',
                        help='Displays the loss on the figures (if applicable).')
    parser.add_argument('--is-posterior', action='store_true',
                        help='Traverses the posterior instead of the prior.')
    parser.add_argument('--store-individuals', action='store_true',
                        help='Flag to tell the program to store individual images used for the traversals-gif.')
    parser.add_argument('-f', '--frequency', type=int, default=0,
                         help='How frequently to store individual posterior traversals for the posterior_traversals.gif (every X epochs). 0 to disable')
    args = parser.parse_args()

    return args


def main(args):
    """Main function for plotting fro pretrained models.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    random_samples = configuration['random_samples'] # ! To toggle using random samples or not
    set_seed(args.seed)
    experiment_name = args.name
    model_dir = os.path.join(RES_DIR, experiment_name)
    meta_data = load_metadata(model_dir)
    model = load_model(model_dir)
    model.eval()  # don't sample from latent: use mean
    # dataset = meta_data['dataset']
    dataset = dataset_config["test_dataset"]
    viz = Visualizer(model=model,
                     model_dir=model_dir,
                     dataset=dataset,
                     max_traversal=args.max_traversal,
                     loss_of_interest='kl_loss_',
                     upsample_factor=args.upsample_factor)
    size = (args.n_rows, args.n_cols)
    
    if configuration['save_dataset'] == True:
        print(f'~~Storing the whole dataset')
        samples = get_dataset(dataset, dataset_config['training_dataset'], configuration['evaluation_path'])
        print(f'LOADED DATASET, SAMPLES ARRAY WITH SHAPE: {samples.shape}')
        if(configuration['get_representations'] == False):
            exit() # Don't exit if we also want to compute the representations!

    if(configuration['get_representations']):
        print(f'~~Getting the representations to compute FactorVAE')
        if(configuration['save_dataset'] == False): # Get the samples
            samples = get_dataset(dataset, dataset_config['training_dataset'])
        viz.get_representation(samples, configuration['evaluation_path'], experiment_name)
        exit()


    # same samples for all plots: sample max then take first `x`data  for all plots
    num_samples = args.n_cols * args.n_rows
    samples = get_samples(dataset, num_samples, idcs=args.idcs, random_ids=random_samples)

    # In case we want to get representations to test our model (i.e. compute a metric)

    is_all_plots = False
    if "all" in args.plots:
        args.plots = [p for p in PLOT_TYPES if p != "all"]
        is_all_plots = True

    for plot_type in args.plots:
        if plot_type == 'generate-samples':
            viz.generate_samples(size=size)
        elif plot_type == 'data-samples':
            viz.data_samples(samples, size=size)
        elif plot_type == "reconstruct":
            viz.reconstruct(samples, size=size)
        elif plot_type == 'traversals':
            viz.traversals(data=samples[0:1, ...] if args.is_posterior else None,
                           n_per_latent=args.n_cols,
                           n_latents=args.n_rows,
                           is_reorder_latents=True)
        elif plot_type == "reconstruct-traverse":
            viz.reconstruct_traverse(samples,
                                     is_posterior=args.is_posterior, #or is_all_plots, # Plot the posteriors when all
                                     n_latents=args.n_rows,
                                     n_per_latent=args.n_cols,
                                     is_show_text=args.is_show_loss)
        elif plot_type == "gif-traversals":
            viz.gif_traversals(samples[:args.n_cols, ...], n_latents=args.n_rows, freq=args.frequency, individuals=args.store_individuals)
        else:
            raise ValueError("Unkown plot_type={}".format(plot_type))


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
