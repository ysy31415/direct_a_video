import os.path
import torch
import decord
decord.bridge.set_bridge('torch')
import random
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

class MyDataset(Dataset):
    def __init__(
            self,
            data_csv: str = None,
            h: int = 512,
            w: int = 512,
            n_sample_frames: int = None,
            sample_start_idx: int = 0,
            sample_frame_stride: int = 5,  # not the final stride
            is_train: bool = True,  # True for training, Fasle for val
            tokenizer=None,
            proportion_empty_prompts: float = 0.1,  # prob of using null text
    ):
        self.is_train = is_train
        self.data_csv = pd.read_csv(data_csv)

        self.w = w
        self.h = h

        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_stride = sample_frame_stride
        self.tokenizer = tokenizer
        self.proportion_empty_prompts = proportion_empty_prompts
        self.iter = 0

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, index):

        batch = self.data_csv.iloc[index]
        cam_motion = torch.tensor(0)
        cam_motion_param = torch.tensor(0)
        video = torch.tensor(0)
        is_input_a_real_video = False

        if self.is_train:
            media_path = batch.get("input")

            if media_path.endswith('.mp4'):  
                vr_input = decord.VideoReader(media_path) # , width=self.w, height=self.h)
                video_length = len(vr_input) 

                sample_frame_indices = self.get_valid_frame_indices(video_length, self.n_sample_frames,
                                                                    self.sample_start_idx, self.sample_frame_stride)

                # read video
                video = vr_input.get_batch(sample_frame_indices)  # [f,w,h,c] 0-255 uint8

                # augment video
                cam_x = 0  if np.random.uniform(0, 1) < 0.33 else np.random.uniform(-1, 1)
                cam_y = 0  if np.random.uniform(0, 1) < 0.33 else np.random.uniform(-1, 1)
                cam_z = 1 if np.random.uniform(0, 1) < 0.33 else 2 ** np.random.uniform(-1, 1)

                cam_motion_param = torch.tensor([cam_x, cam_y, cam_z], dtype=torch.float).unsqueeze(0)

                video, _ = self.aug_with_cam_motion(video, f=self.n_sample_frames, h=self.h, w=self.w,
                                                             cx=cam_x, cy=cam_y, cz=cam_z)

                cam_motion = cam_motion_param

                video = video / 127.5 - 1.0  # [0,255] -> [-1,1]
                video = rearrange(video, "f c h w -> c f h w")
                is_input_a_real_video = True

            elif media_path.endswith(('.png', '.jpg')):  # image format
                raise NotImplementedError
                # image = Image.open(media_path).convert('RGB')
                # video = transforms.ToTensor()(image) # [c,h,w] 0-1 pt float
                # cam_x = 0  if np.random.uniform(0, 1) < 0.33 else np.random.uniform(-1, 1)/2
                # cam_y = 0  if np.random.uniform(0, 1) < 0.33 else np.random.uniform(-1, 1)/2
                # cam_z = 1 if np.random.uniform(0, 1) < 0.33 else 2 ** (np.random.uniform(-1, 1)/2)
                #
                # cam_motion_param = torch.tensor([cam_x, cam_y, cam_z], dtype=torch.float).unsqueeze(0)
                # video, cam_motion = self.aug_with_cam_motion(video, f=self.n_sample_frames, h=self.h, w=self.w,
                #                                              cx=cam_x, cy=cam_y, cz=cam_z)
                # video = rearrange(video, "f c h w -> c f h w")
                # is_input_a_real_video = False

            elif os.path.isdir(media_path):  # a folder containing frames
                raise NotImplementedError
                # media_path = batch.get("input")
                # video_length = min(batch["num_frames"], len(bbox))
                # sample_frame_indices = self.get_valid_frame_indices(video_length, self.n_sample_frames,
                #                                                     self.sample_start_idx, self.sample_frame_stride)
                # video = self.read_imgs_folder_to_video_pt(media_path, sample_frame_indices)


        else:  # val/test mode
            video = 0
            media_path = '__none__'
            cam_x = 0 if np.random.uniform(0, 1)<0.25 else np.random.uniform(-1, 1)
            cam_y = 0 if np.random.uniform(0, 1)<0.25 else np.random.uniform(-1, 1)
            cam_z = 1 if np.random.uniform(0, 1) < 0.25 else 2 ** np.random.uniform(-1, 1)

            cam_motion_param = torch.tensor([cam_x, cam_y, cam_z], dtype=torch.float).unsqueeze(0)
            cam_motion = cam_motion_param

            is_input_a_real_video = False

        text = batch.get('text',"a high quality video") if random.random() > self.proportion_empty_prompts else ""

        self.iter += 1
        return {
            "input": video,
            "text": text,

            "cam_motion_param": cam_motion_param,  #
            "cam_motion": cam_motion,

            "video_name": media_path,
            "is_input_a_real_video": is_input_a_real_video,
        }

    def get_valid_frame_indices(self, video_length, n_sample_frames, sample_start_idx=0, sample_frame_stride=1):
        while True:
            sample_frame_indices = [sample_start_idx + i * sample_frame_stride for i in range(n_sample_frames)]
            if video_length > sample_frame_indices[-1]:
                break
            sample_frame_stride -= 1
        return sample_frame_indices

    def tokenize_captions(self, text):
        if self.is_train and random.random() < self.proportion_empty_prompts:
            text = ""
        inputs = self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        return inputs.input_ids[0]  # inputs.input_ids

    @staticmethod
    def read_imgs_folder_to_video_pt(imgs_folder, sample_frame_indices):
        # img_tensors = [0]*len(sample_frame_indices)
        # img_list = sorted(os.listdir(imgs_folder))
        # for idx in sample_frame_indices:
        #     img = Image.open(os.path.join(imgs_folder, img_list[idx])).convert("RGB")
        #     img_tensor = torch.tensor(np.array(img))
        #     img_tensors[idx] = img_tensor
        # final_tensor = torch.stack(img_tensors).permute(0,3,1,2)
        # return final_tensor

        ## 2X faster version with multi-thread
        img_list = sorted(os.listdir(imgs_folder))
        selected_files = [img_list[idx] for idx in sample_frame_indices]
        first_img = Image.open(os.path.join(imgs_folder, selected_files[0])).convert("RGB")
        img_array = np.array(first_img)
        C, H, W = img_array.shape[2], img_array.shape[0], img_array.shape[1]
        final_np_array = np.empty((len(sample_frame_indices), C, H, W), dtype=np.uint8)

        def load_img_to_array(i, file_name):
            img = Image.open(os.path.join(imgs_folder, file_name)).convert("RGB")
            final_np_array[i] = np.moveaxis(np.array(img), -1, 0)

        with ThreadPoolExecutor() as executor:
            executor.map(load_img_to_array, range(len(selected_files)), selected_files)
        final_tensor = torch.from_numpy(final_np_array)
        return final_tensor

    @staticmethod
    def aug_with_cam_motion(src_media, cx: float = 0.0, cy: float = 0.0, cz: float = 1.0,
          f: int = 8, h: int = 256, w: int = 256):
        """
        该函数使用一个源img/video（src_media）来模拟相机的平移和缩放动作, and generates a video
        Args:
            src_media (torch.Tensor or None): source image/video used to create video, the size can be [h w 3],[f h w 3],[3 h w],[1 3 h w]
            h (int): height of generated video
            w (int): width of generated video
            f (int): num frames of generated video if src_media is an image. If src_media is a video, f will be inherited from it.
            cx (float): x-translation ratio (-1~1), defined as total x-shift of the center between first-last frame / first-frame width
            cy (float): y-translation ratio (-1~1), defined as total y-shift of the center between first-last frame / first-frame height
            cz (float): zoom ratio (0.5~2), defined as scale ratio of last frame / that of the first frame. cz>1 for zoom-in, cz<1 for zoom-out
        Returns:
            generated video in f*c*3*h*w tensor format, and cam_boxes in f*4 tensor format
        """

        cam_boxes = torch.zeros(f, 4)  # 1st box is always [0,0,1,1]
        cam_boxes[:, 0] = torch.linspace(0, cx + (1 - 1 / cz) / 2, f)  # x1
        cam_boxes[:, 1] = torch.linspace(0, cy + (1 - 1 / cz) / 2, f)  # y1
        cam_boxes[:, 2] = torch.linspace(1, cx + (1 + 1 / cz) / 2, f)  # x2
        cam_boxes[:, 3] = torch.linspace(1, cy + (1 + 1 / cz) / 2, f)  # y2

        if src_media is None:
            return None, cam_boxes
        if isinstance(src_media, str) and src_media.endswith(('.png', '.jpg')):  # image
            import torchvision.transforms.functional as TF
            src_frames = torch.stack([TF.to_tensor(Image.open(src_media).convert("RGB"))] * f)
        elif isinstance(src_media, torch.Tensor):
            # [h w 3],[f h w 3],[3 h w],[1 3 h w] -> [f 3 h w]
            src_frames = src_media.unsqueeze(0) if src_media.dim() == 3 else src_media
            src_frames = src_frames.repeat(f, 1, 1, 1) if src_frames.shape[0] == 1 else src_frames
            src_frames = src_frames.permute(0, 3, 1, 2) if src_frames.shape[-1] == 3 else src_frames

            assert src_frames.dim() == 4, 'src_media should be in shape of [f, c, h, w]'
            assert f == src_frames.shape[0], f'f={f} should be the same as src_media.shape[0]={src_frames.shape[0]}'
        else:
            raise TypeError("src_media should be torch.Tensor")

        min_x = torch.min(cam_boxes[:, 0::2])
        max_x = torch.max(cam_boxes[:, 0::2])
        min_y = torch.min(cam_boxes[:, 1::2])
        max_y = torch.max(cam_boxes[:, 1::2])

        normalized_boxes = torch.zeros_like(cam_boxes)
        normalized_boxes[:, 0::2] = (cam_boxes[:, 0::2] - min_x) / (max_x - min_x)
        normalized_boxes[:, 1::2] = (cam_boxes[:, 1::2] - min_y) / (max_y - min_y)

        _, _, src_h, src_w = src_frames.shape

        new_frames = torch.zeros(f, 3, h, w)
        for i, frame in enumerate(src_frames):
            # 定位截取框
            x1, y1, x2, y2 = normalized_boxes[i] * torch.tensor([src_w, src_h, src_w, src_h])
            crop = frame[:, int(y1):int(y2), int(x1):int(x2)].float()
            new_frames[i] = F.interpolate(crop.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

        return new_frames, cam_boxes


# if __name__ == "__main__": ### For debug use only
#     from tqdm import tqdm
#     from utils import draw_bounding_boxes
#
#     val_dataset = MyDataset(
#         data_csv="data/movieshots_v1.csv",
#         h=256,
#         w=256,
#         n_sample_frames=8,
#         sample_frame_stride=5,
#         is_train=False,
#     )
#     val_dataloader = torch.utils.data.DataLoader(
#         val_dataset,
#         shuffle=False,  # check all the training samples
#         # collate_fn=collate_fn,
#         batch_size=1,
#         num_workers=0,
#     )
#     for idx, batch in tqdm(enumerate(val_dataloader, start=0)):
#         print(idx)
#     print('val done!')
#
#     train_dataset = MyDataset(
#         data_csv="data/movieshots_v1.csv",
#         h=256,
#         w=256,
#         n_sample_frames=8,
#         sample_frame_stride=5,
#     )
#     train_dataloader = torch.utils.data.DataLoader(
#         train_dataset,
#         shuffle=False,  # check all the training samples
#         # collate_fn=collate_fn,
#         batch_size=1,
#         num_workers=0,
#     )
#
#     for idx, batch in tqdm(enumerate(train_dataloader, start=0)):
#         pass
#         print(idx, batch['video_name'])
#         v = batch['input']
#
#     print('training done!')
