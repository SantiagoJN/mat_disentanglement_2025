import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from ip_adapter.ip_adapter import ImageProjModel, IPAdapterTRAIN
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

import matplotlib.pyplot as plt
import datetime
import configparser
import numpy as np
import logging
import shutil
import cv2

from torch.utils.tensorboard import SummaryWriter

# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
custom_encoder = True
enable_crop = False

unet_shape = 512
print(f'[INFO] Using unmasked samples at {unet_shape} resolution')

if enable_crop:
    print('Enabled crop of images during training')
else:
    print('Disabled crop of images during training')

# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, tokenizer_2, size=1024, center_crop=True, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, 
                image_root_path="", is_lab=False, extension=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.extension = extension
        self.is_LAB = is_lab
        if self.is_LAB:
            print("[INFO] Using LAB color space")
    
        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform_FVAE = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ])

        self.transform_UNET = transforms.Compose([
            transforms.Resize(unet_shape, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Not normalizing because our FVAE encoder doesn't expect it (https://github.com/tencent-ailab/IP-Adapter/issues/342)
        ])

        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        text = item["text"]
        image_file = item["image"] # Input to FVAE
        
        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        if self.is_LAB: # Transform the image to LAB
            img_cv2 = np.array(raw_image)
            img_lab = cv2.cvtColor(img_cv2,cv2.COLOR_BGR2LAB)
            raw_image = Image.fromarray(img_lab) 
        denoising_image = Image.open(os.path.join(self.image_root_path, image_file))
        
        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform_FVAE(raw_image)

        
        try: # try to get the special image used for the denoising of the UNET
            denoising_file = item["unmasked"]
            raw_denoising = Image.open(f'{self.image_root_path}/../data_unmasked_{unet_shape}{self.extension}/{denoising_file}')
            denoising_im = self.transform_UNET(raw_denoising)
        except KeyError:
            denoising_im = image_tensor

        # random crop
        
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            if enable_crop:
                top = np.random.randint(0, delta_h + 1)
                left = np.random.randint(0, delta_w + 1)
            else:
                top = left = 0
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 
        

        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "image_no_crop": image_tensor,
            "image_denoising": denoising_im,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
            "image_name": image_file
        }
        
    
    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    images_no_crop = torch.stack([example["image_no_crop"] for example in data])
    images_denoising = torch.stack([example["image_denoising"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])
    image_names = [example["image_name"] for example in data]

    return {
        "images": images,
        "images_no_crop": images_no_crop,
        "images_denoising": images_denoising,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        "image_names": image_names
    }
    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

        self.random_ip_tokens = False
        if self.random_ip_tokens:
            print(f'[WARNING] Using random values for text embeddings')
            

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)

        # * Testing: zeroing text embeddings
        if self.random_ip_tokens:
            encoder_hidden_states = torch.randn(encoder_hidden_states.shape)*0.01
            encoder_hidden_states = encoder_hidden_states.to(ip_tokens.device)

        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--denoising_imgs_extension",
        type=str,
        default="",
        required=False,
        help="Extension of the folder containing images used for the denoising UNet; [_fixed, _random, _white] so far",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,              # ! <<<<<<<<
        help="Path to CLIP image encoder",  # ! This should be changed ! 
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--save_epochs",
        type=int,
        default=50,
        help=(
            "Save a checkpoint of the training state every X epochs"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument('--CIELAB', action='store_true',
                        default=False,
                        help="Flag to define if we should convert FVAE input images to CIELAB color space.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    # Get the relevant dimensions from the config file
    config = configparser.RawConfigParser()
    config.read('/mnt/cephfs/home/graphics/sjimenez/IP-Adapter/config.txt')
    configs_dict = dict(config.items('Configs'))
    if "new" in args.image_encoder_path:
        print(f'Getting the relevant dimensions of a new encoder')
        relevants_text = configs_dict['relevant_new'] # Raw text that contains the relevant dimensions' identifiers
    elif "test234" in args.image_encoder_path:
        relevants_text = configs_dict['relevant_old']
    elif "test283" in args.image_encoder_path:
        relevants_text = configs_dict['relevant_old2']
    elif "test278" in args.image_encoder_path:
        relevants_text = configs_dict['relevant_old2_filtered']
    elif "test_architecture_52" in args.image_encoder_path:
        relevants_text = configs_dict['architecture52']
    elif "test_architecture_57" in args.image_encoder_path:
        relevants_text = configs_dict['architecture57']
    else:
        print(f'Warning: unknown image encoder. Relevant dimensions may not be that relevant :)')
        relevants_text = configs_dict['relevant']
    
    relevant_dimensions = np.array([int(num.strip()) for num in relevants_text.split(',')]) # Text to array

    # relevant_dimensions = [3, 6, 7, 11, 13, 17, 19]
    if custom_encoder:
        print(f'[WARNING] Selecting the following dimensions as relevant.\nMake sure this is correct before training\n{relevant_dimensions}')
    else:
        print(f'Using default CLIP image embedder; not selecting any _relevant_ dimensions')

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    writer_dir = f"{args.output_dir}/tensorboard_logs"
    writer = SummaryWriter(writer_dir)

    # Save args into an external json file
    path_to_metadata = f"{args.output_dir}/config.json"
    with open(path_to_metadata, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True) # Save hyperparameters in case we want to check them later
    
    logging.basicConfig(filename=f'{args.output_dir}/run.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Start training IP-Adapter XL!...')
    logging.info(f'Using image encoder in path {args.image_encoder_path}')

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    
    
    if custom_encoder:
        image_encoder = torch.hub.load('/mnt/cephfs/home/graphics/sjimenez/IP-Adapter/disentangling-vae-master2', # hub config location
                        'latent_extractor', 
                        ckpt_path=args.image_encoder_path, # encoder checkpoint
                        source='local')
        emb_dim = len(relevant_dimensions)
    else:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path) # ! FVAE Encoder
        emb_dim = image_encoder.config.projection_dim
    
    
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    #ip-adapter
    num_tokens = 4
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        # clip_embeddings_dim=image_encoder.config.projection_dim, # ! <<<<<<<<<<<<<< 20D?
        clip_embeddings_dim=emb_dim,
        clip_extra_context_tokens=num_tokens, # * any special reason..? --> it may be just because it's the output dimensionality of CLIP
        custom_encoder=custom_encoder
    )
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    print(f"!!!! Accelerator device is {accelerator.device}")
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path,
                              is_lab=args.CIELAB, extension=args.denoising_imgs_extension)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    print(f'[INFO] Training *default IP-Adapter* during {len(train_dataloader)} batches of {args.train_batch_size} size; around {len(train_dataloader)*args.train_batch_size} samples.')
    
    last_saved = None
    global_step = 0
    for epoch in range(1, args.num_train_epochs+1):
        begin = time.perf_counter()
        mean_epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    # ! Images that will be used as ground truth !
                    latents = vae.encode(batch["images_denoising"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor # * https://github.com/huggingface/diffusers/issues/437
                    # print(f'Mean value of latents: {torch.mean(latents)} +- {torch.std(latents)}')
                    
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                with torch.no_grad():
                    if custom_encoder:
                        # ! Images that will be used as conditioning with our FVAE!
                        image_embeds, _ = image_encoder(batch["images_no_crop"].to(accelerator.device, dtype=weight_dtype)) # get the returned *mean*
                        image_embeds = image_embeds[:,relevant_dimensions] # Removing _irrelevant_ dimensions
                    else:
                        image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds # ! VANILLA VERSION
                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)
            
                
                """# * Plotting images to make sure we're introducing proper information into the diffusion process
                print(f'scaling factor: {vae.config.scaling_factor}')
                num_samples = 3
                for s in range(num_samples):
                    print(f'-{batch["image_names"][s]}')
                    counter = 0
                    for d in relevant_dimensions:
                        print(f'\tD{d}: {image_embeds[s][counter]}')
                        counter += 1
                    plt.imshow((((batch["images"][s]).permute(1,2,0)).detach()).cpu())
                    plt.show()
                    # plt.savefig(f"{image_embeds[0]}_images.png")
                    plt.imshow((((batch["images_denoising"][s]).permute(1,2,0)).detach()).cpu())
                    plt.show()
                    # plt.savefig(f"{image_embeds[0]}_images.png")
                exit()
                """


                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat
                        
                # add cond
                add_time_ids = [ # ! Pay special attention to this --> it seems like it is conditioning on crops (I don't know if this is good or bad for us)
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
             

                # print(image_embeds[:4])
                # plt.imshow((((batch["images_no_crop"][0]).permute(1,2,0)).detach()).cpu())
                # plt.show()
                # plt.imshow((((batch["images_no_crop"][1]).permute(1,2,0)).detach()).cpu())
                # plt.show()
                # plt.imshow((((batch["images_no_crop"][2]).permute(1,2,0)).detach()).cpu())
                # plt.show()
                # plt.imshow((((batch["images_no_crop"][3]).permute(1,2,0)).detach()).cpu())
                # plt.show()
                # exit()

                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds)
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    ct = datetime.datetime.now()
                    log_msg = "[{}] Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        ct, epoch, step, load_data_time, time.perf_counter() - begin, avg_loss)
                    logging.info(log_msg)
                    print(log_msg)
                                                              # vv number of steps vv
                writer.add_scalar('batch_loss', avg_loss, (epoch-1)*len(train_dataloader)+step)
                # print(f'=======Writing batch loss no. {(epoch-1)*len(train_dataloader)+step}')
                mean_epoch_loss += avg_loss
            
            global_step += 1

            if global_step % args.save_steps == 0:
                print(f'-----Saving checkpoint at step {global_step}')
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path, safe_serialization=False) 
                if last_saved != None:
                    print(f'Removing last saved model: {last_saved}')
                    shutil.rmtree(last_saved)
                    last_saved = save_path

            begin = time.perf_counter()
        
        
        mean_epoch_loss /= len(train_dataloader)
        writer.add_scalar('epoch_loss', mean_epoch_loss, epoch-1)

        # if epoch % args.save_epochs == 0 and epoch != 0:
        #     save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}")
        #     print(f'Saving in path {save_path}')
        #     accelerator.save_state(save_path, safe_serialization=False) # ! https://github.com/tencent-ailab/IP-Adapter/issues/263

    print(f'~~~~Training Complete~~~~\n...saving checkpoint at the end of training; step={global_step}')
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path, safe_serialization=False)   
                
if __name__ == "__main__":
    main()    
