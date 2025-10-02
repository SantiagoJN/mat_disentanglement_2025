dependencies = ['torch']

import os
import torch

from disvae.utils.modelIO import load_model


def latent_extractor(
    ckpt_path: str
) -> torch.nn.Module:
    
    # Check path exists
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    model = load_model(ckpt_path)
    return model.encoder