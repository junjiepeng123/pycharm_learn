import torch
import numpy as np
import cv2
import logging
import os
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt


def setup_logger(log_path):
    """设置日志"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_seed(seed):
    """设置随机种子"""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2):
    """计算SSIM"""
    # 简化版SSIM计算
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def save_checkpoint(state, filename):
    """保存检查点"""
    torch.save(state, filename)
    logging.info(f"检查点已保存: {filename}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """加载检查点"""
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logging.info(f"加载检查点: {checkpoint_path} (epoch {checkpoint.get('epoch', 'N/A')})")
        return checkpoint.get('epoch', 0), checkpoint.get('best_psnr', 0.0)
    else:
        logging.warning(f"检查点不存在: {checkpoint_path}")
        return 0, 0.0


def tensor_to_image(tensor):
    """将张量转换为图像"""
    # tensor: (C, H, W) or (B, C, H, W)
    if tensor.dim() == 4:
        tensor = tensor[0]

    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image


def visualize_results(lr, hr, sr, save_path=None):
    """可视化结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 如果lr是序列，取中心帧
    if lr.dim() == 5:
        lr = lr[0, lr.shape[1] // 2]

    lr_img = tensor_to_image(lr)
    hr_img = tensor_to_image(hr)
    sr_img = tensor_to_image(sr)

    axes[0].imshow(lr_img)
    axes[0].set_title('LR Input')
    axes[0].axis('off')

    axes[1].imshow(hr_img)
    axes[1].set_title('HR Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(sr_img)
    axes[2].set_title('SR Output')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close(fig)