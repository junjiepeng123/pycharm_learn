import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
import os
from tqdm import tqdm

from config import get_config
from dataset import REDSVideoSuperResolutionDataset
from models import get_model
from utils import setup_logger, set_seed, save_checkpoint, load_checkpoint, visualize_results


def train_epoch(model, train_loader, criterion, optimizer, epoch, config, logger):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for i, batch in enumerate(pbar):
        # 测量数据加载时间
        data_time.update(time.time() - end)

        # 移动到GPU
        lr = batch['lr'].to(config.device)
        hr = batch['hr'].to(config.device)

        # 前向传播
        sr = model(lr)

        # 计算损失
        loss = criterion(sr, hr)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新统计
        batch_size = lr.size(0)
        loss_meter.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)

        # 更新进度条
        pbar.set_postfix({
            'loss': loss_meter.avg,
            'time': batch_time.avg,
            'data': data_time.avg
        })

        # 记录日志
        if (i + 1) % config.log_freq == 0:
            logger.info(f'Epoch [{epoch}/{config.num_epochs}] '
                        f'Batch [{i + 1}/{len(train_loader)}] '
                        f'Loss: {loss_meter.avg:.6f}')

        end = time.time()

    return loss_meter.avg


def validate(model, val_loader, criterion, epoch, config, logger):
    """验证"""
    model.eval()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for i, batch in enumerate(pbar):
            # 移动到GPU
            lr = batch['lr'].to(config.device)
            hr = batch['hr'].to(config.device)

            # 前向传播
            sr = model(lr)

            # 计算损失
            loss = criterion(sr, hr)

            # 计算PSNR和SSIM
            for j in range(sr.size(0)):
                sr_img = sr[j].cpu().numpy()
                hr_img = hr[j].cpu().numpy()

                # 转换为0-255范围
                sr_img = (sr_img * 255).clip(0, 255).astype(np.uint8)
                hr_img = (hr_img * 255).clip(0, 255).astype(np.uint8)

                psnr = calculate_psnr(sr_img, hr_img)
                ssim = calculate_ssim(sr_img, hr_img)

                psnr_meter.update(psnr, 1)
                ssim_meter.update(ssim, 1)

            # 更新损失
            loss_meter.update(loss.item(), lr.size(0))

            # 更新进度条
            pbar.set_postfix({
                'loss': loss_meter.avg,
                'psnr': psnr_meter.avg,
                'ssim': ssim_meter.avg
            })

            # 可视化第一个batch的结果
            if i == 0:
                save_dir = os.path.join(config.exp_dir, 'visualizations')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'epoch_{epoch}.png')
                visualize_results(lr[0], hr[0], sr[0], save_path)

    logger.info(f'Validation - Epoch: {epoch}, Loss: {loss_meter.avg:.6f}, '
                f'PSNR: {psnr_meter.avg:.2f} dB, SSIM: {ssim_meter.avg:.4f}')

    return loss_meter.avg, psnr_meter.avg


def main():
    """主训练函数"""
    # 获取配置
    config = get_config()

    # 设置随机种子
    set_seed(config.seed)

    # 设置日志
    logger = setup_logger(config.log_path)
    logger.info("开始训练")
    logger.info(f"实验配置:\n{config}")

    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    logger.info(f"使用设备: {device}")

    # 创建数据集
    logger.info("加载训练集...")
    train_dataset = REDSVideoSuperResolutionDataset(
        lr_dir=config.train_lr_dir,
        hr_dir=config.train_hr_dir,
        n_frames=config.n_frames,
        patch_size=config.patch_size,
        scale=config.scale,
        img_aug=True,
        is_train=True
    )

    logger.info("加载验证集...")
    val_dataset = REDSVideoSuperResolutionDataset(
        lr_dir=config.val_lr_dir,
        hr_dir=config.val_hr_dir,
        n_frames=config.n_frames,
        patch_size=config.patch_size,
        scale=config.scale,
        img_aug=False,
        is_train=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # 创建模型
    logger.info(f"创建模型: {config.model}")
    model = get_model(
        config.model,
        n_frames=config.n_frames,
        num_features=config.num_features,
        num_blocks=config.num_blocks,
        scale=config.scale
    )
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.L1Loss()  # 视频超分常用L1损失
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )

    # 恢复训练
    start_epoch = 0
    best_psnr = 0.0

    if config.resume:
        start_epoch, best_psnr = load_checkpoint(
            config.resume, model, optimizer
        )

    # 训练循环
    logger.info("开始训练循环")
    for epoch in range(start_epoch, config.num_epochs):
        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, epoch, config, logger
        )

        # 验证
        if (epoch + 1) % config.val_freq == 0:
            val_loss, val_psnr = validate(
                model, val_loader, criterion, epoch, config, logger
            )

            # 保存最佳模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_model_path = os.path.join(
                    config.exp_dir, f'best_model_epoch_{epoch + 1}_psnr_{val_psnr:.2f}.pth'
                )
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'psnr': val_psnr,
                    'best_psnr': best_psnr
                }, best_model_path)

        # 保存检查点
        if (epoch + 1) % config.save_freq == 0:
            checkpoint_path = os.path.join(
                config.exp_dir, f'checkpoint_epoch_{epoch + 1}.pth'
            )
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'psnr': val_psnr if 'val_psnr' in locals() else 0.0,
                'best_psnr': best_psnr
            }, checkpoint_path)

        # 调整学习率
        scheduler.step()

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch + 1}: LR = {current_lr:.6f}')

    logger.info(f"训练完成! 最佳PSNR: {best_psnr:.2f} dB")


class AverageMeter:
    """计算和存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 导入PSNR和SSIM计算函数
from utils import calculate_psnr, calculate_ssim

if __name__ == '__main__':
    import numpy as np

    main()