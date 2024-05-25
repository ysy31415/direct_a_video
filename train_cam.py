#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and


import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any

import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import random
from einops import rearrange, repeat
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)

from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from src.unet_3d import MyUNet3DConditionModel as UNet3DConditionModel
from val_cam import log_validation

import warnings
import itertools

warnings.filterwarnings("ignore", category=FutureWarning)

if is_wandb_available():
    pass

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.17.0.dev0")
"""
    When debug code, comment out the above line and using ldm env; 
    When run code, uncomment the above line and using controlnet env
"""
logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="cerspense/zeroscope_v2_576w",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--h", type=int, default=512, help=(
        "The height for input images, all the images in the train/validation dataset will be resized to this"" resolution"), )
    parser.add_argument("--w", type=int, default=512, help=(
        "The width for input images, all the images in the train/validation dataset will be resized to this"" resolution"), )

    parser.add_argument("--train_batch_size",
                        type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--n_sample_frames", type=int, default=8, help="num of frames to sample for each video")
    parser.add_argument("--sample_frame_stride", type=int, default=5, help="sample frame interval")

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.", )
    parser.add_argument(
        "--checkpointing_interval",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_interval`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--scale_lr",
                        action="store_true",
                        default=False,
                        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
                        )
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'), )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles",
                        type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.", )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help=(
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."), )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--allow_tf32",
                        action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    parser.add_argument("--mixed_precision",
                        type=str, default=None, choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
                        ),
                        )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    parser.add_argument(
        "--train_data_csv", type=str, default="data/example.csv",
        help="csv file of train data, cna be generated using make_csv.py."
    )
    parser.add_argument(
        "--val_data_csv", type=str, default="data/example.csv",
        help="csv file of train data, cna be generated using make_csv.py."
    )


    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.05,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )

    parser.add_argument(
        "--validation_interval",
        type=int,
        default=1000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
        ),
    )

    parser.add_argument("--val_sanity_check", action="store_true", help="run val at first iter")

    parser.add_argument("--tracker_project_name", type=str, default="my_demo", help=(
        "The `project_name` argument passed to Accelerator.init_trackers for"
        " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
    ),
                        )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    assert 0 <= args.proportion_empty_prompts <= 1, "`--proportion_empty_prompts` must be in the range [0, 1]."
    assert os.path.exists(os.path.join(os.getcwd(), args.train_data_csv)), f"{args.train_data_csv} does not exist!"
    assert os.path.exists(os.path.join(os.getcwd(), args.val_data_csv)), f"{args.val_data_csv} does not exist!"
    assert args.h % 8 == 0, "`--h` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
    assert args.w % 8 == 0, "`--w` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."

    return args


def main(args):
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # backup src files
    if accelerator.is_main_process:
        from utils import save_project_src_files
        save_project_src_files(["src/*.py", "./*.py", "./*.sh"], os.path.join(args.output_dir, 'code'))

    def _load_models(): pass

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,
                                        local_files_only=True, )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    ## image_processor is the preprocessor for vae encoder inputs and decoder outputs, NOT encoding or decoding between pix and latnet!
    # vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    # if args.resume_from_checkpoint is None:  # train from original model
    unet_config = UNet3DConditionModel.load_config(args.pretrained_model_name_or_path, subfolder='unet')
    unet_config['attention_type'] = "cross_temp"  # set attention_type to "cross_temp" so that cam_embedder and cam_module can be added
    unet = UNet3DConditionModel.from_config(unet_config)
    print(f"Loading unet weights from original pretrained unet: {args.pretrained_model_name_or_path}!")
    unet_orig = UNet3DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='unet',
                                                     revision=args.revision, local_files_only=True, )
    unet.load_state_dict(unet_orig.state_dict(), strict=False)
    del unet_orig
    unet.init_new_attn_layer_weights()  # optional: init cam_module weights from native self-attn layers
    unet_cross_attention_dim = unet.cross_attention_dim


    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer',
                                              revision=args.revision, local_files_only=True, )
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder',
                                                 revision=args.revision, local_files_only=True, )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",
                                                    local_files_only=True, )  # e-pred
    noise_scheduler_prediction_type = noise_scheduler.config.prediction_type  # "epsilon" or "v-pred"

    @torch.no_grad()
    def D(latents_):
        if latents_.dim() == 5:
            return \
            vae.decode(rearrange(latents_, "b c f h w -> (b f) c h w") / vae.config.scaling_factor, return_dict=False)[
                0]
        else:
            return vae.decode(latents_ / vae.config.scaling_factor, return_dict=False)[0]

    @torch.no_grad()
    def E(pixels_):
        b = pixels_.shape[0]
        if pixels_.dim() == 5:  # video
            pixels = rearrange(pixels_, "b c f h w -> (b f) c h w")
        else:  # input.dim() == 4: # image
            raise NotImplementedError("Currently not implemented for 4D image")
        latents_ = vae.encode(pixels.to(dtype=train_dtype)).latent_dist.sample() * vae.config.scaling_factor
        latents_ = rearrange(latents_, "(b f) c h w -> b c f h w", b=b)
        return latents_

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            unet_ = models[0]
            unet_.save_pretrained(os.path.join(output_dir, "unet"))
            weights.clear()  # empty the weights so that they are not saved again

        def load_model_hook(models, input_dir):
            unet_ = models[0]
            from safetensors.torch import load_file
            unet_ckpt = load_file(os.path.join(input_dir, "unet", "diffusion_pytorch_model.safetensors"))
            unet_.load_state_dict(unet_ckpt, strict=False)
            del unet_ckpt

            models.clear()  # empty the models so that they are not loaded again

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    def _config_optimizer(): pass

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.requires_grad_(False)
    # unet_trainable_layers.requires_grad_(True) # the same as below
    unet_trainable_params = []
    for name, params in unet.named_parameters():
        if 'fuser' in name or 'position_net' in name:  # cam module has 'fuser', cam embedder has 'position_net'
            print(f"Trainable layer: {name}") if accelerator.is_main_process else None
            params.requires_grad = True
            unet_trainable_params.append(params)
    unet.train()

    if accelerator.is_main_process:
        print(f"Unet total params: {sum(p.numel() for p in unet.parameters())}")
        print(f"Unet trainable params: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.")
            unet.enable_xformers_memory_efficient_attention()
            # controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        print("gradient_checkpointing enabled!")
        unet.enable_gradient_checkpointing()
        # controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    # if accelerator.unwrap_model(controlnet).dtype != torch.float32:
    #     raise ValueError(
    #         f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
    #     )
    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = itertools.chain(unet_trainable_params)
    # params_to_optimize = itertools.chain(unet_trainable_params, controlnet_trainable_params)
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    print(f"learning_rate: {args.learning_rate}, weight_decay:{args.adam_weight_decay}")

    def _prepare_dataset():
        pass

    from src.dataset import MyDataset
    with accelerator.main_process_first():
        train_dataset = MyDataset(
            data_csv=args.train_data_csv,
            h=args.h,
            w=args.w,
            n_sample_frames=args.n_sample_frames,
            sample_frame_stride=args.sample_frame_stride,
            proportion_empty_prompts=args.proportion_empty_prompts,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            # collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    """
    When the gradient_accumulation_steps option is used, the max_train_steps will be automatically calculated 
    according to the number of epochs and the length of the training dataset
    """
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # scheduler can be obtained from diffusers.optimization

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    train_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        train_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        train_dtype = torch.bfloat16

    def _move_model_to_device():
        pass

    # Move vae, unet image_encoder and text_encoder to device and cast to train_dtype
    vae.to(accelerator.device, dtype=train_dtype)
    text_encoder.to(accelerator.device, dtype=train_dtype)
    if accelerator.mixed_precision == "fp16":
        '''
        when using fp16, will encounter the ValueError: Attempting to unscale FP16 gradients,
        to solve this, keep the trainable part in fp32, and cast the rest in fp16.
        see:  https://github.com/huggingface/peft/issues/341
        '''
        unet.to(accelerator.device)
        for param in unet.parameters():
            if param.requires_grad:   # 如果参数是可训练的，将其转换为float
                param.data = param.data.float()
            else:         # 否则，将其转换为half
                param.data = param.data.half()
    else:  # "bf16" or "fp32"
        unet.to(accelerator.device, dtype=train_dtype)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # use tracker_config.pop("validation_prompt") when tensorboard cannot handle list types for config
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    def _resume_ckpt(): pass
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            ## (Recommended first) load full states
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(path.split("-")[1])  # for example, checkpoint-1000
            accelerator.print(f"Successfully resuming from checkpoint {path}")

            ## (Secondary option) use following code if you failed loading optimizer,
            ## often happens after you changed model structure, this loads unet only
            # from safetensors.torch import load_file
            # unet_ckpt = load_file(os.path.join(args.resume_from_checkpoint, "unet", "diffusion_pytorch_model.safetensors"))
            # unet.load_state_dict(unet_ckpt, strict=False)
            # del unet_ckpt
            # accelerator.print(f"Warning: only unet ckpt is resumed from {args.resume_from_checkpoint}")

            initial_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def _train_loop():
        pass

    val_sanity_check_flag = True
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):  # accelerator.autocast()

                # Convert images to latent space
                inputs = batch["input"].to(dtype=train_dtype)
                b, c, f, h_latent, w_latent = inputs.shape[0:5]
                latents = E(inputs)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                # Sample a random timestep for each image
                timesteps = torch.randint(400, noise_scheduler.config.num_train_timesteps, (b,),
                                          device=accelerator.device).long()

                # forward diffusion process, add noise to latent, except 1st frame
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    is_input_a_real_video = batch["is_input_a_real_video"]
                    text_prompt = batch["text"]
                    text_tokens = tokenizer(text_prompt, max_length=tokenizer.model_max_length, padding="max_length",
                                            truncation=True, return_tensors="pt").input_ids.to(accelerator.device)

                    text_emb = text_encoder(text_tokens).last_hidden_state  # [1,77,768|1024]
                    text_emb = text_emb.repeat_interleave(repeats=f,
                                                          dim=0)  # [f,77,768|1024], repeat text emb on frame dim (actually impl on batch dim)

                    # get cam motion,
                    cam_motion_param = batch["cam_motion_param"]
                    if torch.any(cam_motion_param):
                        cam_motion = batch["cam_motion"].to(dtype=train_dtype)
                    else:
                        cam_motion = None

                    attn_kwargs = {}
                    attn_kwargs["cam"] = {"cam_motion": cam_motion,}

                # Unet forwarding
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_emb,
                                  cross_attention_kwargs=attn_kwargs).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler_prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler_prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler_prediction_type}! ")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")  # by default, loss is L2 loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = params_to_optimize
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    # save ckpt
                    if global_step % args.checkpointing_interval == 0 or \
                       os.path.exists(os.path.join(args.output_dir, "ckpt", "ckpt_now")): # ckpt immediately by creating a new file "ckpt_now" in ckpt dir
                        save_path = os.path.join(args.output_dir, "ckpt", f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        if os.path.exists(os.path.join(args.output_dir, "ckpt", "ckpt_now")):
                            os.remove(os.path.join(args.output_dir, "ckpt", "ckpt_now"))

                    # run validation
                    if global_step % args.validation_interval == 0 or (args.val_sanity_check and val_sanity_check_flag):
                        def _val(): pass
                        print(f"Running validation code at global_step {global_step}")
                        unet.eval()
                        image_logs = log_validation(vae, text_encoder, tokenizer, unet, noise_scheduler,
                                                    args, accelerator, global_step, train_dtype, val_sanity_check_flag)
                        unet.train()
                        val_sanity_check_flag = False


            accelerator.wait_for_everyone()
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    def _end_training_():
        pass

    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
