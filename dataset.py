import torch
import torch.utils.data as data
import os
import numpy as np
from torch.utils.data import Dataset
import random
import cv2
import logging


class REDSVideoSuperResolutionDataset(Dataset):
    """用于视频超分辨率任务的REDS数据集"""

    def __init__(self, lr_dir, hr_dir, n_frames=5, patch_size=256, scale=4,
                 img_aug=True, is_train=True):
        self.n_frames = n_frames
        self.patch_size = patch_size
        self.scale = scale
        self.img_aug = img_aug
        self.is_train = is_train

        # 存储每个视频的帧路径
        self.lr_frame_paths = []
        self.hr_frame_paths = []
        self.video_start_indices = []
        self.video_names = []

        # 获取所有视频文件夹
        video_folders = sorted([f for f in os.listdir(hr_dir)
                                if os.path.isdir(os.path.join(hr_dir, f))])
        #video_folders=['002', '003', '004', '005', '007', '008', '009',……]
        #hr_dir= './Datasets/VSR/REDS/train/GT'
        for vid_folder in video_folders:
            hr_vid_path = os.path.join(hr_dir, vid_folder)
            lr_vid_path = os.path.join(lr_dir, vid_folder)

            if not os.path.exists(lr_vid_path):
                logging.warning(f"低分辨率文件夹不存在: {lr_vid_path}")
                continue

            # 获取排序后的帧文件名
            hr_frames = sorted([f for f in os.listdir(hr_vid_path)
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
            #hr_frames=['00000000.png', '00000001.png', '00000002.png'
            lr_frames = sorted([f for f in os.listdir(lr_vid_path)
                                if f.endswith(('.png', '.jpg', '.jpeg'))])

            # 确保帧数一致
            if len(hr_frames) != len(lr_frames):
                logging.warning(f"视频{vid_folder}的HR和LR帧数不一致: {len(hr_frames)} vs {len(lr_frames)}")
                continue

            # 记录视频起始索引
            self.video_start_indices.append(len(self.hr_frame_paths))
            #video_start_indices=[0,100,200]
            self.video_names.append(vid_folder)
            #vid_folder='002'
            # 添加完整路径
            for hr_frame, lr_frame in zip(hr_frames, lr_frames):
                self.hr_frame_paths.append(os.path.join(hr_vid_path, hr_frame))
                self.lr_frame_paths.append(os.path.join(lr_vid_path, lr_frame))

        logging.info(f"数据集加载完成: {len(video_folders)}个视频, 共{len(self.hr_frame_paths)}帧")

    def __len__(self):
        return len(self.hr_frame_paths)

    def augment_img(self, img, mode=0):
        """数据增强"""
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(img)
        elif mode == 2:
            return np.rot90(img)
        elif mode == 3:
            return np.flipud(np.rot90(img))
        elif mode == 4:
            return np.rot90(img, k=2)
        elif mode == 5:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 6:
            return np.rot90(img, k=3)
        elif mode == 7:
            return np.flipud(np.rot90(img, k=3))
        return img

    def get_frame_indices(self, idx):
        """获取帧索引，处理边界条件"""
        m = self.n_frames // 2

        # 找到当前帧所属的视频的起始和结束索引
        video_start_idx = max([i for i in self.video_start_indices if i <= idx])
        video_end_idx = min([i for i in self.video_start_indices if i > idx],
                            default=len(self.hr_frame_paths)) - 1
        #就是找当前帧属于哪个视频
        # 获取T帧的索引
        indices = [idx + i for i in range(-m, m + 1)]

        # 处理边界情况（使用反射填充）
        processed_indices = []
        for i in indices:
            if i < video_start_idx:
                i = video_start_idx + (video_start_idx - i)
            elif i > video_end_idx:
                i = video_end_idx - (i - video_end_idx)
            processed_indices.append(i)

        return processed_indices

    def __getitem__(self, idx):
        # 获取帧索引
        indices = self.get_frame_indices(idx)

        # 读取和准备帧
        lr_frames = []
        hr_frames = []

        # 随机数据增强模式（仅在训练时）
        aug_mode = np.random.randint(0, 8) if self.img_aug and self.is_train else 0

        for frame_idx in indices:
            # 读取低分辨率图像
            lr_img = cv2.imread(self.lr_frame_paths[frame_idx], cv2.IMREAD_COLOR)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
            lr_img = lr_img.astype(np.float32) / 255.0

            # 读取高分辨率图像
            hr_img = cv2.imread(self.hr_frame_paths[frame_idx], cv2.IMREAD_COLOR)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            hr_img = hr_img.astype(np.float32) / 255.0

            # 应用相同的数据增强
            lr_img = self.augment_img(lr_img, aug_mode)
            hr_img = self.augment_img(hr_img, aug_mode)

            # 转换为CHW格式
            lr_img = lr_img.transpose(2, 0, 1)  # C, H, W
            hr_img = hr_img.transpose(2, 0, 1)  # C, H, W

            lr_frames.append(lr_img)
            hr_frames.append(hr_img)

        # 训练时随机裁剪，测试/验证时不裁剪
        if self.is_train:
            # 随机裁剪
            _, lr_h, lr_w = lr_frames[0].shape

            # 确保patch_size不超过图像尺寸
            crop_lr_size = min(self.patch_size // self.scale, lr_h, lr_w)
            crop_hr_size = crop_lr_size * self.scale

            # 随机选择裁剪位置
            rnd_h = random.randint(0, max(0, lr_h - crop_lr_size))
            rnd_w = random.randint(0, max(0, lr_w - crop_lr_size))

            # 裁剪所有帧
            lr_cropped = []
            hr_cropped = []

            for lr_frame, hr_frame in zip(lr_frames, hr_frames):
                lr_crop = lr_frame[:, rnd_h:rnd_h + crop_lr_size, rnd_w:rnd_w + crop_lr_size]
                hr_crop = hr_frame[:,
                          rnd_h * self.scale:(rnd_h + crop_lr_size) * self.scale,
                          rnd_w * self.scale:(rnd_w + crop_lr_size) * self.scale]

                lr_cropped.append(lr_crop)
                hr_cropped.append(hr_crop)

            lr_frames = lr_cropped
            hr_frames = hr_cropped

        # 堆叠为序列 (T, C, H, W)
        lr_sequence = np.stack(lr_frames, axis=0)
        hr_sequence = np.stack(hr_frames, axis=0)

        # 转换为张量
        lr_sequence = torch.from_numpy(lr_sequence).float()
        hr_sequence = torch.from_numpy(hr_sequence).float()

        m = self.n_frames // 2

        return {
            'lr': lr_sequence,  # T, C, H, W
            'hr': hr_sequence[m],  # C, H, W (中心帧)
            'indices': indices,
            'center_idx': indices[m]
        }