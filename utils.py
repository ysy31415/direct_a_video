import torch
import torch.nn.functional as F
from einops import rearrange
import decord
decord.bridge.set_bridge('torch')
import numpy as np



def load_bbox(path, f):
    bbox = torch.from_numpy(np.load(path))
    indices = np.linspace(0, len(bbox)-1, f).astype(int)
    return bbox[indices]

import re
def prompt_to_segment(prompt):
    '''
    input prompt = "a (cute dog) and a wolf* running on the ( beach)"
    returns "a cute dog and a wolf running on the beach"  AND   ["cute  dog", "wolf", "beach"]
    '''
    combined_pattern = r'\(\s*([^)]+?)\s*\)|\b(\w+)\*'
    # Find all matches for bracketed and starred fields
    matches = re.findall(combined_pattern, prompt)
    # Process matches to flatten the result and remove empty strings
    combined_fields = [' '.join(match).strip() for match in matches if any(match)]
    # Clean the prompt by removing just the parentheses and asterisk characters, keeping the words
    cleaned_prompt = re.sub(r'[()]*', '', prompt)  # Remove parentheses
    cleaned_prompt = re.sub(r'\*', '', cleaned_prompt)  # Remove asterisk
    cleaned_prompt = ' '.join(cleaned_prompt.split())  # Normalize spaces
    return cleaned_prompt, combined_fields


def read_video(video_path,frame_indices=None,h=256,w=256,f=8):
    # returns [1,c,f,h,w]
    vr = decord.VideoReader(video_path, width=w, height=h)
    total_frames = len(vr)
    assert total_frames > f, f'video {video_path}\'s length {total_frames} is smaller than f={f}!'
    frame_indices = frame_indices or torch.linspace(0, total_frames - 1, f).long()
    video = vr.get_batch(frame_indices)  # [f,w,h,c] uint8
    video = video / 127.5 - 1.0  # [0,255] -> [-1,1]
    video = rearrange(video, "f h w c -> 1 c f h w")
    return video


import glob
import os
import shutil


def save_project_src_files(file_paths, save_dir):
    '''
    Backup files to save_dir, support regex
    Example usage:
    save_project_src_files(["src/*.py", "./*.py", "./*.sh"], './code')
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file_path in file_paths:
        matched_files = glob.glob(file_path)
        for src_path in matched_files:
            relative_path = os.path.relpath(src_path)
            dest_path = os.path.join(save_dir, relative_path)
            dest_dir = os.path.dirname(dest_path)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            shutil.copy2(src_path, dest_path)


import cv2


def draw_bounding_boxes(video, bbox=None):
    '''
    draw bbox on video
    （1）bbox: torch tensor with the size of n*f*4, value range = 0~1, f=num_frames, n=num_objs
    （2）video： a List of np.numpy, whose length is num of frames, each numpy should be h*w*3
    '''

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255,255,0), (255,0,255), (0,255,255)]
    annotated_video = []
    if isinstance(video, torch.Tensor):
        video = video.squeeze().cpu()
        video = (255*(video+1)/2).clamp(0, 255).byte() if video.min() < -0.5 else (255*video).clamp(0, 255).byte()
        if video.shape[0] == 3:
            video = [video[:, i, :, :].permute(1, 2, 0).numpy() for i in range(video.shape[1])]
        elif video.shape[1] == 3:
            video = [video[i, :, :, :].permute(1, 2, 0).numpy() for i in range(video.shape[0])]
    if bbox is None:
        print("No bounding box provided, returning original video")
        return [frame.copy() for frame in video]

    if isinstance(bbox, torch.Tensor):
        if bbox.dim() == 2:  # [f=8, 4]
            bbox = bbox.unsqueeze(0).tolist()  # # List:[1, f=8, 4]
    for frame_idx, frame in enumerate(video):
        frame = frame.copy()  # Copy the frame to not alter the original video
        frame_height, frame_width, _ = frame.shape
        for obj_idx, obj_frames in enumerate(bbox):
            color = colors[obj_idx % len(colors)]
            x1, y1, x2, y2 = [int(coord * (frame_width if i % 2 == 0 else frame_height)) for i, coord in
                              enumerate(obj_frames[frame_idx])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        annotated_video.append(frame)

    return annotated_video

def calculate_optical_flow_from_boxes(boxes, h=256, w=256):
    '''
    given boxes (fx4 pt.tensor), calculate optical flow 2x(f-1)xhxw
    '''
    boxes.clip_(0, 1)
    def interpolate_flow(box1, box2, h, w):
        # 计算box顶点的位移
        dx1 = (box2[0] - box1[0]) * w
        dy1 = (box2[1] - box1[1]) * h
        dx2 = (box2[2] - box1[2]) * w
        dy2 = (box2[3] - box1[3]) * h
        # 初始化光流场
        flow = torch.zeros((2, h, w))
        for y in range(int(box1[1] * h), int(box1[3] * h)):
            for x in range(int(box1[0] * w), int(box1[2] * w)):
                wx = (x - box1[0] * w) / ((box1[2] - box1[0]) * w)
                wy = (y - box1[1] * h) / ((box1[3] - box1[1]) * h)

                # 插值计算光流
                flow[0, y, x] = wx * dx2 + (1 - wx) * dx1
                flow[1, y, x] = wy * dy2 + (1 - wy) * dy1

        return flow

    n = boxes.shape[0]
    optical_flow = torch.zeros((2, n - 1, h, w))

    for i in range(n - 1):
        box1, box2 = boxes[i], boxes[i + 1]
        flow = interpolate_flow(box1, box2, h, w)
        optical_flow[:, i, :, :] = flow

    return optical_flow  # 2*(f-1)*h*w


def generate_mask_from_bbox(bbox, f, h, w):
    # box should be [f,4]
    assert isinstance(bbox, torch.Tensor), 'bbox should be a tensor'
    x1, y1, x2, y2 = (bbox * torch.tensor([w, h, w, h]).type_as(bbox)).int().unbind(-1)

    row_idx = torch.arange(h,device=bbox.device).view(1, 1, -1, 1)
    col_idx = torch.arange(w,device=bbox.device).view(1, 1, 1, -1)

    row_mask = (row_idx >= y1.view(f, -1, 1, 1)) & (row_idx < y2.view(f, -1, 1, 1))
    col_mask = (col_idx >= x1.view(f, -1, 1, 1)) & (col_idx < x2.view(f, -1, 1, 1))

    combined_mask = row_mask & col_mask

    # 确保所有bbox内的值都置为1
    mask = combined_mask.any(dim=1, keepdim=True).float()

    mask = (F.interpolate(mask, size=(h, w), mode='bilinear') > 0.01).float()  # [f,1,h,w]
    mask = mask.permute(1, 0, 2, 3)
    return mask.type_as(bbox)


import imageio
def save_tensor_as_mp4(tensor, filename):
    tensor = tensor.detach().float().cpu()
    tensor = tensor * 0.5 + 0.5 # [-1,1] -> [0,1]
    tensor = (255*tensor.squeeze(0).permute(1, 2, 3, 0)).clamp(0,255).to(torch.uint8).numpy()
    images = [tensor[i] for i in range(tensor.shape[0])]
    imageio.mimsave(filename, images, format='mp4', fps=8)



def visualize_bbox(bbox, height=320, width=512, frames=24):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    colors = ['r', 'g', 'b']
    fig, ax = plt.subplots(figsize=(width/80, height/80))
    ax.imshow(np.ones((height, width, 3)))
    for j, box in enumerate(bbox):
        centroids = []
        for i in range(frames):
            x1, y1, x2, y2 = box[i] * np.array([width, height, width, height])
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor=colors[j], facecolor='none')
            ax.add_patch(rect)
            centroids.append([(x1+x2)/2, (y1+y2)/2])

        for i in range(len(centroids)-1):
            ax.arrow(centroids[i][0], centroids[i][1], centroids[i+1][0]-centroids[i][0], centroids[i+1][1]-centroids[i][1], color=colors[j], head_width=10)
    plt.tight_layout()
    plt.show()