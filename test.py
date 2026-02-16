import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse
import logging

from config import get_config
from dataset import REDSVideoSuperResolutionDataset
from models import get_model
from utils import setup_logger, load_checkpoint, calculate_psnr, calculate_ssim


def test_model(model, test_loader, config, logger):
    """测试模型"""
    model.eval()

    # 初始化统计
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    # 逐视频测试
    video_psnrs = {}
    video_ssims = {}

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for batch_idx, batch in enumerate(pbar):
            # 移动到GPU
            lr = batch['lr'].to(config.device)
            hr = batch['hr'].to(config.device)

            # 前向传播
            sr = model(lr)

            # 计算每个样本的PSNR和SSIM
            for i in range(sr.size(0)):
                # 转换为numpy
                sr_img = sr[i].cpu().numpy()
                hr_img = hr[i].cpu().numpy()

                # 转换为0-255范围
                sr_img = np.transpose(sr_img, (1, 2, 0))
                hr_img = np.transpose(hr_img, (1, 2, 0))

                sr_img = (sr_img * 255).clip(0, 255).astype(np.uint8)
                hr_img = (hr_img * 255).clip(0, 255).astype(np.uint8)

                # 计算PSNR和SSIM
                psnr = calculate_psnr(sr_img, hr_img)
                ssim = calculate_ssim(sr_img, hr_img)

                total_psnr += psnr
                total_ssim += ssim
                total_samples += 1

                # 更新进度条
                avg_psnr = total_psnr / total_samples
                avg_ssim = total_ssim / total_samples
                pbar.set_postfix({
                    'PSNR': f'{avg_psnr:.2f}',
                    'SSIM': f'{avg_ssim:.4f}'
                })

            # 保存部分结果图像
            if batch_idx == 0:
                save_visual_results(lr, hr, sr, config.result_dir)

    # 计算平均值
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples

    logger.info(f"测试结果 - 平均PSNR: {avg_psnr:.2f} dB, 平均SSIM: {avg_ssim:.4f}")

    # 保存结果到文件
    result_file = os.path.join(config.result_dir, 'test_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"测试配置:\n")
        f.write(f"模型: {config.model}\n")
        f.write(f"帧数: {config.n_frames}\n")
        f.write(f"缩放比例: {config.scale}\n")
        f.write(f"\n测试结果:\n")
        f.write(f"平均PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"平均SSIM: {avg_ssim:.4f}\n")
        f.write(f"测试样本数: {total_samples}\n")

    return avg_psnr, avg_ssim


def save_visual_results(lr, hr, sr, save_dir):
    """保存可视化结果"""
    # 只保存前3个样本
    for i in range(min(3, lr.size(0))):
        # 转换张量为图像
        lr_img = lr[i].cpu().numpy()
        hr_img = hr[i].cpu().numpy()
        sr_img = sr[i].cpu().numpy()

        # 取LR序列的中心帧
        if lr_img.ndim == 4:  # (T, C, H, W)
            lr_img = lr_img[lr_img.shape[0] // 2]

        # 转换为0-255范围
        lr_img = np.transpose(lr_img, (1, 2, 0))
        hr_img = np.transpose(hr_img, (1, 2, 0))
        sr_img = np.transpose(sr_img, (1, 2, 0))

        lr_img = (lr_img * 255).clip(0, 255).astype(np.uint8)
        hr_img = (hr_img * 255).clip(0, 255).astype(np.uint8)
        sr_img = (sr_img * 255).clip(0, 255).astype(np.uint8)

        # 保存图像
        cv2.imwrite(os.path.join(save_dir, f'sample_{i}_lr.png'),
                    cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f'sample_{i}_hr.png'),
                    cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f'sample_{i}_sr.png'),
                    cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))

        # 保存对比图
        save_comparison(lr_img, hr_img, sr_img,
                        os.path.join(save_dir, f'sample_{i}_comparison.png'))


def save_comparison(lr, hr, sr, save_path):
    """保存对比图"""
    # 调整大小以便拼接
    h, w = hr.shape[:2]
    lr_resized = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

    # 创建对比图像
    comparison = np.concatenate([lr_resized, hr, sr], axis=1)

    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'LR (bicubic)', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'HR (GT)', (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'SR (Ours)', (2 * w + 10, 30), font, 1, (255, 255, 255), 2)

    cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='视频超分辨率测试')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--test_lr_dir', type=str,
                        default='./Datasets/VSR/REDS/test/bicubic',
                        help='测试集低分辨率图像目录')
    parser.add_argument('--test_hr_dir', type=str,
                        default='./Datasets/VSR/REDS/test/GT',
                        help='测试集高分辨率图像目录')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU ID')

    args = parser.parse_args()

    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置日志
    logger = setup_logger('./logs/test.log')
    logger.info(f"使用设备: {device}")

    # 创建测试数据集
    logger.info("加载测试集...")
    test_dataset = REDSVideoSuperResolutionDataset(
        lr_dir=args.test_lr_dir,
        hr_dir=args.test_hr_dir,
        n_frames=5,  # 使用5帧
        patch_size=256,
        scale=4,
        img_aug=False,
        is_train=False
    )

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 测试时batch_size=1
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 创建模型（需要从检查点推断模型参数）
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_state = checkpoint['model_state_dict']

    # 推断模型参数
    n_frames = 5  # 默认值
    scale = 4  # 默认值

    # 尝试从检查点推断
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        n_frames = model_config.get('n_frames', 5)
        scale = model_config.get('scale', 4)
        model_name = model_config.get('model', 'SimpleVSR')
    else:
        # 简单推断
        model_name = 'SimpleVSR'

    logger.info(f"创建模型: {model_name}")
    model = get_model(
        model_name,
        n_frames=n_frames,
        num_features=64,
        num_blocks=10,
        scale=scale
    )

    # 加载模型权重
    model.load_state_dict(model_state)
    model = model.to(device)

    # 创建结果目录
    result_dir = './test_results'
    os.makedirs(result_dir, exist_ok=True)

    # 测试
    logger.info("开始测试...")
    avg_psnr, avg_ssim = test_model(model, test_loader,
                                    type('Config', (), {
                                        'device': device,
                                        'model': model_name,
                                        'n_frames': n_frames,
                                        'scale': scale,
                                        'result_dir': result_dir
                                    }), logger)

    logger.info(f"测试完成! PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")


if __name__ == '__main__':
    main()