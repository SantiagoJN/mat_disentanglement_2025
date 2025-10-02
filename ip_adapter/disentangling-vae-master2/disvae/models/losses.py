"""
Module containing all vae losses.
"""
import abc
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

# import cv2
import random

import numpy as np
import OpenEXR as exr
import Imath
from PIL import Image
from PIL import ImageChops
import matplotlib.pyplot as plt
import torchvision.models as models

from utils.helpers import get_config_section
CONFIG_FILE = "hyperparam.ini"
configuration = get_config_section([CONFIG_FILE], "Testing")

from .discriminator import Discriminator
from disvae.utils.math import (log_density_gaussian, log_importance_weight_matrix,
                               matrix_log_density_gaussian)


LOSSES = ["VAE", "betaH", "betaB", "factor", "btcvae"]
RECON_DIST = ["bernoulli", "laplace", "gaussian", "crossentropy", "huber", "smooth", "perceptual"]

# TO-DO: clean n_data and device
def get_loss_f(loss_name, **kwargs_parse):
    """Return the correct loss function given the argparse arguments."""
    kwargs_all = dict(rec_dist=kwargs_parse["rec_dist"],
                      steps_anneal=kwargs_parse["reg_anneal"])
    if loss_name == "betaH":
        return BetaHLoss(beta=kwargs_parse["betaH_B"], **kwargs_all)
    elif loss_name == "VAE":
        return BetaHLoss(beta=1, **kwargs_all)
    elif loss_name == "betaB":
        return BetaBLoss(C_init=kwargs_parse["betaB_initC"],
                         C_fin=kwargs_parse["betaB_finC"],
                         gamma=kwargs_parse["betaB_G"],
                         **kwargs_all)
    elif loss_name == "factor":
        return FactorKLoss(kwargs_parse["device"],
                           gamma=kwargs_parse["factor_G"],
                           beta=kwargs_parse["factor_B"],
                           c=kwargs_parse["factor_C"],
                           warmup_c_epochs=kwargs_parse["warmup_C_epochs"],
                           max_c=kwargs_parse["max_C"],
                           growth_scale=kwargs_parse["growth_scale"],
                           epochs=kwargs_parse["epochs"],
                           disc_kwargs=dict(latent_dim=kwargs_parse["latent_dim"]),
                           optim_kwargs=dict(lr=kwargs_parse["lr_disc"], betas=(0.5, 0.9)),
                           max_dim=kwargs_parse["max_dim"], max_error=kwargs_parse["max_error"], 
                           use_GECO=kwargs_parse["use_GECO"],
                           warmup_epochs=kwargs_parse["warmup_epochs"],
                           max_beta=kwargs_parse["max_beta"],
                           cycle=kwargs_parse["cycle"],
                           recon_factor=kwargs_parse["recon_factor"],
                           kl_func=kwargs_parse["kl_func"],
                           **kwargs_all)
    elif loss_name == "btcvae":
        return BtcvaeLoss(kwargs_parse["n_data"],
                          alpha=kwargs_parse["btcvae_A"],
                          beta=kwargs_parse["btcvae_B"],
                          gamma=kwargs_parse["btcvae_G"],
                          **kwargs_all)
    else:
        assert loss_name not in LOSSES
        raise ValueError("Uknown loss : {}".format(loss_name))


class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    record_loss_every: int, optional
        Every how many steps to recorsd the loss.

    rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction distribution istribution of the likelihood on the each pixel.
        Implicitely defines the reconstruction loss. Bernoulli corresponds to a
        binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
        corresponds to L1.

    steps_anneal: nool, optional
        Number of annealing steps where gradually adding the regularisation.
    """

    def __init__(self, record_loss_every=50, rec_dist="bernoulli", steps_anneal=0):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).

        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).

        is_train : bool
            Whether currently in train mode.

        storer : dict
            Dictionary in which to store important variables for vizualisation.

        kwargs:
            Loss specific arguments
        """

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if not is_train or self.n_train_steps % self.record_loss_every == 1:
            storer = storer
        else:
            storer = None

        return storer


class BetaHLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    def __init__(self, beta=4, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)
        loss = rec_loss + anneal_reg * (self.beta * kl_loss)

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


class BetaBLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C.

    C_fin : float, optional
        Final annealed capacity C.

    gamma : float, optional
        Weight of the KL divergence term.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
    """

    def __init__(self, C_init=0., C_fin=20., gamma=100., **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin

    def __call__(self, data, recon_data, latent_dist, is_train, storer, first_iteration=False, **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)

        C = (linear_annealing(self.C_init, self.C_fin, self.n_train_steps, self.steps_anneal)
             if is_train else self.C_fin)

        loss = rec_loss + self.gamma * (kl_loss - C).abs()

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


class FactorKLoss(BaseLoss):
    """
    Compute the Factor-VAE loss as per Algorithm 2 of [1]

    Parameters
    ----------
    device : torch.device

    gamma : float, optional
        Weight of the TC loss term. `gamma` in the paper.

    discriminator : disvae.discriminator.Discriminator

    optimizer_d : torch.optim

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """

    def __init__(self, device,
                 gamma=10.,
                 beta=1.,
                 c=1.,
                 max_c=0,
                 warmup_c_epochs=-1,
                 growth_scale=1e-20,
                 epochs=0,
                #  warmup=100,
                 disc_kwargs={},
                 optim_kwargs=dict(lr=5e-5, betas=(0.5, 0.9)),
                 max_dim=0, max_error=0, use_GECO=False,
                 warmup_epochs=-1,
                 max_beta=1,
                 cycle=False,
                 recon_factor=1.0,
                 kl_func=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.beta = beta
        self.C = c
        self.device = device
        self.discriminator = Discriminator(**disc_kwargs).to(self.device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), **optim_kwargs)
        self.kl_func = kl_func
        
        self.blob_mask = readEXR(configuration['blob_mask_path'])
        self.sphere_mask = readEXR(configuration['sphere_mask_path'])
        self.blob_mask = ((torch.from_numpy(np.array(self.blob_mask))).to(device)) * -1.0
        self.sphere_mask = ((torch.from_numpy(np.array(self.sphere_mask))).to(device)) * -1.0

        self.vgg = models.vgg16(pretrained=True).features.eval()
        self.vgg = self.vgg.to(self.device)

        self.recon_factor = recon_factor

        self.use_GECO = use_GECO
        # * Part of the implementation of GECO
        if self.use_GECO:
            self.max_dim = max_dim
            self.current_dim = max_dim 
            gates = np.ones(max_dim) # At first they are all open
            self.gates = torch.tensor(gates, device=self.device)
            self.max_error = max_error
        else:
            self.current_dim = 0 
            self.max_error = 0

        # self.warming_up = False
        # self.cycle = cycle
        # self.max_beta = max_beta # maximum value of beta that we handle (default=1)
        # if warmup_epochs != -1:
        #     print(f'[INFO] Doing warmup for Factor_B weight during {warmup_epochs} epochs')
        #     self.warmup_step = self.max_beta/warmup_epochs
        #     self.beta = -self.warmup_step # to make it start at 0 after first update
        #     self.warming_up = True
        
        # self.warming_up_C = False
        # self.max_c = max_c
        # if warmup_c_epochs != -1:
        #     print(f'[INFO] Doing warmup for C weight during {warmup_c_epochs} epochs')
        #     self.warmup_C_step = self.max_c/warmup_c_epochs
        #     self.C = -self.warmup_C_step # to make it start at 0 after first update
        #     self.warming_up_C = True
        
        
        # ! Precompute the beta and C values during all the training phase
        self.beta_values = [max_beta] * epochs
        warmup_epochs = int(warmup_epochs)
        if warmup_epochs != -1:
            t_values = np.linspace(0, warmup_epochs, warmup_epochs)
            self.beta_values = max_beta * (np.exp(growth_scale * t_values) - 1) / (np.exp(growth_scale * warmup_epochs) - 1)
            self.beta_values = self.beta_values.tolist()
            print(self.beta_values[:10])
            if cycle:
                num_cycles = int(np.ceil(epochs / warmup_epochs))
                self.beta_values = self.beta_values * num_cycles
                self.beta_values = self.beta_values[:epochs]
            else:
                self.beta_values = self.beta_values + ([max_beta] * int(epochs-warmup_epochs))
        
        self.C_values = [max_c] * epochs
        warmup_c_epochs = int(warmup_c_epochs)
        if warmup_c_epochs != -1:
            t_values = np.linspace(0, warmup_c_epochs, warmup_c_epochs)
            self.C_values = max_c * (np.exp(growth_scale * t_values) - 1) / (np.exp(growth_scale * warmup_c_epochs) - 1)
            self.C_values = self.C_values.tolist()
            print(self.C_values[:10])
            if cycle:
                num_cycles = int(np.ceil(epochs / warmup_c_epochs))
                self.C_values = self.C_values * num_cycles
                self.C_values = self.C_values[:epochs]
            else:
                self.C_values = self.C_values + ([max_c] * int(epochs-warmup_c_epochs))
        
        self.global_iteration = -1
        self.beta = -1
        self.C = -1


    def __call__(self, *args, **kwargs):
        raise ValueError("Use `call_optimize` to also train the discriminator")

    def compute_test_loss(self, data, model, optimizer, storer):
        print("WARNING: COMPUTING LOSS WITHOUT TRAINING DISCRIMINATOR (use `call_optimize` for this)")
        # raise ValueError("Use `call_optimize` to also train the discriminator")

        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]
        # Factor VAE Loss
        recon_batch, latent_dist, latent_sample1 = model(data1)
        rec_loss = _reconstruction_loss(data1, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)

        kl_loss = _kl_normal_loss(*latent_dist, storer)

        d_z = self.discriminator(latent_sample1)
        # We want log(p_true/p_false). If not using logisitc regression but softmax
        # then p_true = exp(logit_true) / Z; p_false = exp(logit_false) / Z
        # so log(p_true/p_false) = logit_true - logit_false
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        # with sigmoid (not good results) should be `tc_loss = (2 * d_z.flatten()).mean()`

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if model.training else 1)
        vae_loss = rec_loss + kl_loss + anneal_reg * self.gamma * tc_loss

        return vae_loss
    
    def call_optimize(self, data, model, optimizer, storer, normals=None, labels=None, objective=None, first_iteration=False):
        storer = self._pre_call(model.training, storer)

        if first_iteration:
            self.global_iteration += 1
            self.beta = self.beta_values[self.global_iteration]
            self.C = self.C_values[self.global_iteration]
            # print(f'B = {self.beta}, C = {self.C}')

            if storer is not None:
                storer['beta'].append(self.beta)
                storer['C'].append(self.C)
            # print(f'beta: {self.beta}')
            # if storer is not None:
            #     storer['beta'].append(self.beta)
            #     storer['C'].append(self.C)
            # if self.warming_up: # * BETA warming up
            #     self.beta += self.warmup_step # time to update beta!
            #     if self.beta > self.max_beta: # It already achieved its maximum
            #         if self.cycle: # set the beta value back to 0, and continue
            #             self.beta = 0
            #         else:
            #             self.beta = self.max_beta
            #             self.warming_up = False # Not warming up anymore, beta sticks to 1
            # if self.warming_up_C: # * C warming up
            #     self.C += self.warmup_C_step # time to update beta!
            #     if self.C > self.max_c: # It already achieved its maximum
            #         if self.cycle: # set the beta value back to 0, and continue
            #             self.C = 0
            #         else:
            #             self.C = self.max_c
            #             self.warming_up_C = False # Not warming up anymore, beta sticks to 1

        # factor-vae split data into two batches. In the paper they sample 2 batches
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]
        if normals != None:
            normals = normals.split(half_batch_size)
            normals1 = normals[0]
            normals2 = normals[1]
        else:
            normals1 = normals2 = None
        objective = objective.split(half_batch_size)
        objective1 = objective[0]
        objective2 = objective[1]

        # Factor VAE Loss
        if self.use_GECO: # * Part of GECO implementation
            latent_dist = model.encoder(data1) # Before applying gates
            # print(f'BEFORE: {type(latent_dist)}, {type(latent_dist[0])}, {type(latent_dist[0][0])}, {latent_dist[0].dtype} --> \n{latent_dist}')
            # print(f'Type of gates: {type(self.gates)}, -> {self.gates}')
            latent_dist = ((latent_dist[0] * self.gates).to(torch.float32), (latent_dist[1] * self.gates).to(torch.float32))
            # print(f'AFTER: {type(latent_dist)}, {type(latent_dist[0])}, {type(latent_dist[0][0])} --> \n{latent_dist}')
            # exit()
            latent_sample1 = model.reparameterize(*latent_dist)
            recon_batch = model.decoder(latent_sample1, normals1)

        else:
            recon_batch, latent_dist, latent_sample1 = model(data1, normals1)
        
        """
        # Visualizing how are the reconstructions converging!
        print(f'Visualizing how the model reconstructs the input images! (imshow function around losses.py:400)')
        # TODO: Igual se podría hacer una visualización del training con esto, sacando la reconstrucción de la misma muestra para cada epoch, y hacer un gif
        cv2.imshow("objective",objective1[0].permute(1,2,0).cpu().numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("reconstruction",recon_batch[0].permute(1,2,0).cpu().detach().numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        
        rec_loss = _reconstruction_loss(objective1, recon_batch, # !Changed "data" by "objective1", since now the reconstruction objective may not coincide with the input data
                                            storer=storer,
                                            distribution=self.rec_dist,
                                            labels=labels,
                                            blob_mask=self.blob_mask,
                                            sphere_mask=self.sphere_mask,
                                            vgg = self.vgg) * self.recon_factor # * Factor to weigh reconstruction loss

        if self.use_GECO: # * Modify or not the gates depending on rec_loss
#            print(f'[INFO] GECO method, rec_loss = {rec_loss}, max_error = {self.max_error}, with Z={self.current_dim}')
            if rec_loss < self.max_error: # We have room to reduce Z
                self.current_dim -= 1
                assert self.current_dim > 0, "[ERROR] Latent dimension set to 0!"
            else: # We are at the beginning or we have reduced Z too much
                self.current_dim = min(self.current_dim + 1, self.max_dim)
            # After modifying the current dim, update the gates
            ones = np.ones(self.current_dim)
            zeros = np.zeros(self.max_dim-self.current_dim)
            gates = np.concatenate((ones, zeros))
            self.gates = torch.tensor(gates, device=self.device)

        # kl_loss = KL(q(z|f), p(z))
        kl_loss = _kl_normal_loss(*latent_dist, storer, kl_func=int(self.kl_func))

        d_z = self.discriminator(latent_sample1)
        # We want log(p_true/p_false). If not using logisitc regression but softmax
        # then p_true = exp(logit_true) / Z; p_false = exp(logit_false) / Z
        # so log(p_true/p_false) = logit_true - logit_false
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        # with sigmoid (not good results) should be `tc_loss = (2 * d_z.flatten()).mean()`

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if model.training else 1)
        vae_loss = rec_loss + self.beta * torch.abs(kl_loss - self.C) + anneal_reg * self.gamma * tc_loss

        if storer is not None:
            storer['loss'].append(vae_loss.item())
            storer['tc_loss'].append(tc_loss.item())
            storer['Z'].append(self.current_dim)
            storer['max_error'].append(self.max_error) # To have this value saved (although it is constant)

        if not model.training:
            # don't backprop if evaluating
            return vae_loss

        # Compute VAE gradients
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)

        # Discriminator Loss
        # Get second sample of latent distribution
        latent_sample2 = model.sample_latent(data2)
        z_perm = _permute_dims(latent_sample2).detach()
        d_z_perm = self.discriminator(z_perm)

        # Calculate total correlation loss
        # for cross entropy the target is the index => need to be long and says
        # that it's first output for d_z and second for perm
        ones = torch.ones(half_batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_perm, ones))
        # with sigmoid would be :
        # d_tc_loss = 0.5 * (self.bce(d_z.flatten(), ones) + self.bce(d_z_perm.flatten(), 1 - ones))

        # TO-DO: check ifshould also anneals discriminator if not becomes too good ???
        #d_tc_loss = anneal_reg * d_tc_loss

        # Compute discriminator gradients
        self.optimizer_d.zero_grad()
        d_tc_loss.backward()

        # Update at the end (since pytorch 1.5. complains if update before)
        optimizer.step()
        self.optimizer_d.step()

        if storer is not None:
            storer['discrim_loss'].append(d_tc_loss.item())

        return vae_loss


class BtcvaeLoss(BaseLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, n_data, alpha=1., beta=6., gamma=1., is_mss=True, **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling

    def __call__(self, data, recon_batch, latent_dist, is_train, storer,
                 latent_sample=None, first_iteration=False):
        storer = self._pre_call(is_train, storer)
        batch_size, latent_dim = latent_sample.shape

        rec_loss = _reconstruction_loss(data, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                             latent_dist,
                                                                             self.n_data,
                                                                             is_mss=self.is_mss)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)

        # total loss
        loss = rec_loss + (self.alpha * mi_loss +
                           self.beta * tc_loss +
                           anneal_reg * self.gamma * dw_kl_loss)

        if storer is not None:
            storer['loss'].append(loss.item())
            storer['mi_loss'].append(mi_loss.item())
            storer['tc_loss'].append(tc_loss.item())
            storer['dw_kl_loss'].append(dw_kl_loss.item())
            # computing this for storing and comparaison purposes
            _ = _kl_normal_loss(*latent_dist, storer)

        return loss


def perceptual_loss(vgg, reconstructed, original):
    features_reconstructed = vgg(reconstructed)
    features_original = vgg(original)
    loss = F.mse_loss(features_reconstructed, features_original)
    return loss


def _reconstruction_loss(data, recon_data, distribution="bernoulli", storer=None, labels=None, blob_mask=None, sphere_mask=None, vgg=None):
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).

    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).

    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.

    storer : dict
        Dictionary in which to store important variables for vizualisation.

    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size, n_chan, height, width = recon_data.size()
    is_colored = n_chan == 3
    loss_mode = configuration['recon_loss_mode']   # ! The "mode" in which the bernoulli loss is computed
                    # ! 1: naive - compute BCE and sum all the components
                    # ! 2: sampled - loss is a sum of a subset of all the BCE values
                    # ! 3: normalized - loss is divided by the resolution
                    # ! 4: post_masked - applying the mask only when computing the loss
    num_samples = configuration['num_samples'] 
    factor = int(height * width)
        

    if distribution == "bernoulli":
        if(loss_mode == 1):
            loss = F.binary_cross_entropy(recon_data, data, reduction="sum") # Bernoulli
        elif(loss_mode == 2): # -> doesn't work
            losses_separated = F.binary_cross_entropy(recon_data, data, reduction="none")
            samples_height = random.sample(range(height), k=num_samples) # Get #num_samples random samples within the range of the height
            samples_width = random.sample(range(width), k=num_samples) # "" width
            loss = torch.sum(losses_separated[:, :, samples_height, samples_width])
        elif(loss_mode == 3):
            loss = F.binary_cross_entropy(recon_data, data, reduction="mean")
            loss = loss * factor
        elif(loss_mode == 4):
            loss = F.binary_cross_entropy(recon_data_aux, data_aux, reduction="sum")
        else:
            print("[ERROR] Recon_loss loss mode is invalid.")
    elif distribution == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
        loss = F.mse_loss(recon_data * 255, data * 255, reduction="sum") / 255
    elif distribution == "laplace":
        # loss in [0,255] space but normalized by 255 to not be too big but
        # multiply by 255 and divide 255, is the same as not doing anything for L1
        loss = F.l1_loss(recon_data, data, reduction="sum")
        loss = loss * 3  # emperical value to give similar values than bernoulli => use same hyperparam
        loss = loss * (loss != 0)  # masking to avoid nan
    elif distribution == "crossentropy":
        loss = F.cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "huber":
        loss = F.huber_loss(recon_data, data, reduction="sum")
    elif distribution == "smooth":
        loss = F.smooth_l1_loss(recon_data, data, reduction="sum")
        loss = loss * (loss != 0) 
    elif distribution == "perceptual":
        loss = perceptual_loss(vgg, recon_data, data) # perceptual loss
    else:
        assert distribution not in RECON_DIST
        raise ValueError("Unkown distribution: {}".format(distribution))

    loss = loss / batch_size

    if storer is not None:
        storer['recon_loss'].append(loss.item())

    return loss


def _kl_normal_loss(mean, logvar, storer=None, kl_func=0):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)

    storer : dict
        Dictionary in which to store important variables for vizualisation.
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    if kl_func == 0:
        total_kl = latent_kl.sum()
    else:
        total_kl = torch.linalg.norm(latent_kl, ord=kl_func) # L2 norm --> encourage distribution of KL over dimensions of Z
    # print(f'KL:\narray {latent_kl.shape}: {latent_kl}\nsum: {latent_kl.sum()}\nL2: {torch.linalg.norm(latent_kl, ord=2)}')

    if storer is not None:
        storer['kl_loss'].append(total_kl.item())
        for i in range(latent_dim):
            storer['kl_loss_' + str(i)].append(latent_kl[i].item())

    return total_kl


def _permute_dims(latent_sample):
    """
    Implementation of Algorithm 1 in ref [1]. Randomly permutes the sample from
    q(z) (latent_dist) across the batch for each of the latent dimensions (mean
    and log_var).

    Parameters
    ----------
    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        shape : (batch_size, latent_dim).

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).

    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


# Batch TC specific
# TO-DO: test if mss is better!
def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx

def readEXR(filename):
    """Read color + depth data from EXR image file.
    
    Parameters
    ----------
    filename : str
        File path.
        
    Returns
    -------
    img : RGB or RGBA image in float32 format. Each color channel
          lies within the interval [0, 1].
          Color conversion from linear RGB to standard RGB is performed
          internally. See https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
          for more information.
          
    Z : Depth buffer in float32 format or None if the EXR file has no Z channel.
    """
    
    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    
    channelData = dict()
    
    # convert all channels in the image to numpy arrays
    for c in header['channels']:
        # print(c)
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)
        
        channelData[c] = C
    
    # print(header['channels'])
    colorChannels = ['albedo.R', 'albedo.G', 'albedo.B', 'albedo.A'] if 'albedo.A' in header['channels'] else ['albedo.R', 'albedo.G', 'albedo.B']
    img = np.concatenate([channelData[c][...,np.newaxis] for c in colorChannels], axis=2)
    
    # linear to standard RGB
    img[..., :3] = np.where(img[..., :3] <= 0.0031308,
                            12.92 * img[..., :3],
                            1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055)
    
    # sanitize image to be in range [0, 1]
    img = np.where(img < 0.0, 0.0, np.where(img > 1.0, 1, img))
    
    #Z = None if 'Z' not in header['channels'] else channelData['Z']
    
    img = Image.fromarray(np.uint8(img*255))
    img = img.resize((256,256))
    img = img.convert('1')

    return img