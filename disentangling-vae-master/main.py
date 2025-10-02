
import argparse
import logging
import sys
import os
from configparser import ConfigParser

from torch import optim
import torch
import numpy as np
import time

from disvae import init_specific_model, Trainer, Evaluator
from disvae.utils.modelIO import save_model, load_model, load_metadata, save_metadata
from disvae.models.losses import LOSSES, RECON_DIST, get_loss_f
from disvae.models.vae import MODELS
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_, FormatterNoDuplicate)
from utils.visualize import GifTraversalsTraining

from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"]="0"


CONFIG_FILE = "hyperparam.ini"
path_config = get_config_section([CONFIG_FILE], "Paths")
dataset_config = get_config_section([CONFIG_FILE], "Datasets")
RES_DIR = path_config['results_dir']
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]


def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name', type=str,
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--use-normals', action='store_true',
                         default=default_config['use_normals'],
                         help='Uses normal maps of samples.')
    general.add_argument('--CIELAB', action='store_true',
                         default=default_config['use_lab'],
                         help='Uses CIELAB color space.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')
    general.add_argument('-f', '--frequency', type=int, default=default_config['frequency'],
                         help='How frequently to store individual traversals during learning (every X epochs). 0 to disable')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=EXPERIMENTS,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('-t', '--triplets-name', type=str,
                          default=default_config['triplets_name'],
                          help='Name of the triplets file we use for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')
    training.add_argument('--skip', type=int, default=default_config['skip'],
                          help='How many deconvolutions to skip in the decoder')
    training.add_argument('--ratio', type=float, default=default_config['ratio'],
                          help='Percentage of the image that will be substituted by black dots')
    training.add_argument('--drop', type=int, default=default_config['drop'],
                          help='Once every what times do we _drop_ pixels')



    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                       choices=RECON_DIST,
                       help="Form of the likelihood ot use for each pixel.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")
    
    model.add_argument('--train_folder',
                       default=dataset_config['training_dataset'],
                       help="Folder where the training samples are located.")
    
    model.add_argument('--from_checkpoint',
                       default=default_config['from_checkpoint'],
                       help="Location of the checkpoint we want to load.")

    # Loss Specific Options
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--factor-B', type=float,
                        default=default_config['factor_B'],
                        help="Weight of the KL term (for avoiding posterior collapse).")
    factor.add_argument('--factor-C', type=float,
                        default=default_config['factor_C'],
                        help="Target KL divergence value (to encourage distributing info; src: understanding disentangling in b-vae).")
    factor.add_argument('--max-C', type=float,
                        default=default_config['max_C'],
                        help="Maximum value of C factor in our model.")
    factor.add_argument('--warmup-C-epochs', type=float,
                        default=default_config['warmup_C_epochs'],
                        help="Number of epochs that Factor_C takes to go from 0 to max_c. If -1, warmup is disabled.")
    factor.add_argument('--warmup-epochs', type=float,
                        default=default_config['warmup_epochs'],
                        help="Number of epochs that Factor_B takes to go from 0 to 1. If -1, warmup is disabled.")
    factor.add_argument('--max-beta', type=float,
                        default=default_config['max_beta'],
                        help="Maximum value of Beta factor in our model.")
    factor.add_argument('--cycle', action='store_true',
                        default=default_config['cycle'],
                        help="Flag to define if we perform cycling over Beta values.")
    factor.add_argument('--lr-disc', type=float,
                        default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')
    factor.add_argument('--recon_factor', type=float,
                        default=default_config['recon_factor'],
                        help='Factor used as weight for the recon_loss factor in the vae loss.')
    factor.add_argument('--kl_func', type=int,
                        default=default_config['kl_func'],
                        help='Function used for kl loss. If 0, sum() is used. Else, L\{N\}-norm is used.')
    factor.add_argument('--growth_scale', type=float,
                        default=default_config['growth_scale'],
                        help='Term to define exponential slope. If undefined, a linear function will be used.')
    factor.add_argument('--big', action='store_true',
                        default=default_config['big'],
                        help="Flag to define if we are using 'big' samples (1024x1024) or not (only implemented for full-masked-serrano so far).")
    # * For the GECO implementation
    factor.add_argument('--max-dim', type=int,
                        default=default_config['max_dim'],
                        help="Initial maximum dimensions (GECO implementation).")
    factor.add_argument('--max-error', type=int,
                        default=default_config['max_error'],
                        help="Maximum error handled (GECO implementation).")
    factor.add_argument('--use-GECO', type=int,
                        default=default_config['use_GECO'],
                        help="Whether or not use the GECO implementation.")
    

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Learning options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')
    
    args = parser.parse_args(args_to_parse)
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            # update all common sections first
            model, dataset = args.experiment.split("_")
            common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
            update_namespace_(args, common_data)
            common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
            update_namespace_(args, common_model)

        try:
            experiments_config = get_config_section([CONFIG_FILE], args.experiment)
            update_namespace_(args, experiments_config)
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e  # only reraise if didn't use common section

    return args


def main(args):
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    start_time = time.time()
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = torch.device('cuda:0')
    print(f'----Using {torch.cuda.get_device_name(0)}----')
    exp_dir = os.path.join(RES_DIR, args.name)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))

    writer_root = path_config['runs_root']
    writer_dir = f"{writer_root}/{args.dataset}/{args.name}"
    writer = SummaryWriter(writer_dir)
    

    if not args.is_eval_only:

        create_safe_directory(exp_dir, logger=logger)
        print(f'Saving metadata...')
        save_metadata(metadata=vars(args), directory=exp_dir)

        if args.use_normals:
            print(f'[INFO] Using normal maps for the decoder')

        if args.loss == "factor":
            #logger.info("FactorVae needs 2 batches per iteration. To replicate this behavior while being consistent, we double the batch size and the the number of epochs.")
            #args.batch_size *= 2
            #args.epochs *= 2
            logger.info("FactorVAE does not duplicate the epochs anymore")

        # PREPARES DATA
        train_loader = get_dataloaders(args.dataset,
                                       batch_size=args.batch_size,
                                       train_fld=args.train_folder,
                                       logger=logger,
                                       triplets_name=args.triplets_name,
                                       ratio=args.ratio,
                                       drop=args.drop,
                                       is_LAB=args.CIELAB)
        
        logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))

        compute_test_losses = False
        if args.dataset == "serrano" and compute_test_losses: # Load the test dataset to compute its loss during training
            test_loader = get_dataloaders("serranotest",
                                       batch_size=args.batch_size,
                                       logger=logger)
            test_loss_f = get_loss_f(args.loss,
                            n_data=len(test_loader.dataset),
                            device=device,
                            **vars(args))
            compute_test_losses = False #! To tell the training process to store test losses
            if compute_test_losses:
                logger.info("Test {} with {} samples".format("serranotest", len(test_loader.dataset)))

        else:
            test_loader = None # Else, just don't declare it
            test_loss_f = None

        # PREPARES MODEL
        args.img_size = get_img_size(args.dataset)  # stores for metadata
        if args.from_checkpoint == "Undefined":
            model = init_specific_model(args.model_type, args.img_size, args.latent_dim, args.batch_size, args.use_normals, args.skip)
        else:
            model = load_model(f"results/{args.from_checkpoint}", True, "model.pt", skip=2  )
            print(f'[OK] Successfully loaded the checkpoint at {args.from_checkpoint}')
        logger.info('Num parameters in model: {}'.format(get_n_param(model)))

        # TRAINS
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        model = model.to(device)  # make sure trainer and viz on same device
        # gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir, store_freq=args.frequency)
        gif_visualizer = None # Don't save anything, training gif is not so useful and slows down the training :/
        loss_f = get_loss_f(args.loss,
                            n_data=len(train_loader.dataset),
                            device=device,
                            **vars(args))
        trainer = Trainer(model, optimizer, loss_f,
                          device=device,
                          logger=logger,
                          save_dir=exp_dir,
                          is_progress_bar=not args.no_progress_bar,
                          gif_visualizer=gif_visualizer,
                          writer_dir=writer_dir,
                          writer=writer)
        trainer(train_loader,
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every,
                test_loader=test_loader,
                compute_test_losses=compute_test_losses,
                test_loss_f=test_loss_f)
        
        args.trainDS = dataset_config['training_dataset'] # To save it in the config.json

    print('Skipping evaluation metrics due to memory constraints')

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f'[SUCCESS] Training done in {hours} hours, {minutes} minutes and {seconds:.2f} seconds.')
        
    # SAVE MODEL AND EXPERIMENT INFORMATION
    args.trainingTime = f"{hours} hours and {minutes} minutes"
    save_model(trainer.model, exp_dir, metadata=vars(args))

    writer.close() # Make sure it is closed properly 


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if args.checkpoint_every == 30:
        args.checkpoint_every = np.floor(args.epochs/10)
    print(f'Saving checkpoints every {args.checkpoint_every} epochs.')
    print(f'[INFO] Dataset selected: {args.dataset} : {args.train_folder}')
    main(args)
