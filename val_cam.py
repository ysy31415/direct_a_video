
import argparse
import logging
import math
import os
import shutil
from pathlib import Path
import imageio
import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger

from diffusers import (
    AutoencoderKL,
    DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
)

from diffusers.utils import is_wandb_available
from src.t2v_pipeline import TextToVideoSDPipeline
from src.dataset import MyDataset
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__)

@torch.no_grad()
def log_validation(vae, text_encoder, tokenizer,unet, scheduler,
                   args, accelerator, global_step, val_dtype, val_sanity_check_flag=True):
    logger.info("Running validation... ")

    pipeline = TextToVideoSDPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        scheduler=DPMSolverMultistepScheduler.from_config(scheduler.config),
        # scheduler=DDIMScheduler.from_config(pipeline.scheduler.config)
    )

    # print(pipeline.scheduler.config.prediction_type)  # >>> "epsilon"
    pipeline.set_progress_bar_config(disable=False)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)

    val_img_save_dir = os.path.join(args.output_dir, 'val')
    os.makedirs(val_img_save_dir, exist_ok=True)

    val_dataset = MyDataset(data_csv=args.val_data_csv,
                            h=args.h,
                            w=args.w,
                            n_sample_frames=args.n_sample_frames,
                            sample_frame_stride=args.sample_frame_stride,
                            is_train=False,
                            )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        # collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    neg_prompt = ["blurry image, disfigured, bad anatomy, low resolution, deformed body features, poorly drawn face, bad composition"]
    pos_prompt = ", aesthetic masterpiece, highly detailed, 4K resolution, high quality, "

    for idx, batch in enumerate(val_dataloader):
        prompt = batch["text"]
        cam_motion_param = batch["cam_motion_param"]
        if torch.any(cam_motion_param):
            cam_motion = batch["cam_motion"].to(device=accelerator.device, dtype=val_dtype)
            cam_x, cam_y, cam_z = [i.item() for i in cam_motion_param.squeeze()]
        else: # batch["cam_motion"]=torch.tensor([0]), means no cam_motion
            cam_motion = None

        print(f"Validating {idx} - {prompt}")
        out = pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            cam_motion=cam_motion,
            cam_off_t=0.85,
            cam_cfg=True,
            num_frames=args.n_sample_frames,
            width=args.w,
            height=args.h,
            guidance_scale=8,
            generator=generator, #[torch.Generator(device="cuda").manual_seed(seed)],
        ).frames

        #  out = draw_bounding_boxes(out, boxes)
        if cam_motion is not None:
            imageio.mimsave(os.path.join(val_img_save_dir, f"{global_step:08d}-{idx:02d}-{prompt[:50]}-"
                                                       f"cam=({cam_x:.2f},{cam_y:.2f},{cam_z:.2f}).mp4"), out, fps=8)
        else:
            imageio.mimsave(os.path.join(val_img_save_dir, f"{global_step:08d}-{idx:02d}-{prompt[:50]}.mp4"), out, fps=8)

        if val_sanity_check_flag: # run once in 1st iter
            break

    pipeline.enable_fuser(True)  # enable fuser again, otherwise the cam module will be disabled during next training
    torch.cuda.empty_cache()
    return


