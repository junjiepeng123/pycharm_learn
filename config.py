import argparse
import os
from datetime import datetime


def get_config():
    parser = argparse.ArgumentParser(description='Video Super Resolution Training')

    # 数据参数
    parser.add_argument('--train_lr_dir', type=str,
                        default='./Datasets/VSR/REDS/train/bicubic',
                        help='训练集低分辨率图像目录')
    parser.add_argument('--train_hr_dir', type=str,
                        default='./Datasets/VSR/REDS/train/GT',
                        help='训练集高分辨率图像目录')
    parser.add_argument('--val_lr_dir', type=str,
                        default='./Datasets/VSR/REDS/val/bicubic',
                        help='验证集低分辨率图像目录')
    parser.add_argument('--val_hr_dir', type=str,
                        default='./Datasets/VSR/REDS/val/GT',
                        help='验证集高分辨率图像目录')
    parser.add_argument('--test_lr_dir', type=str,
                        default='./Datasets/VSR/REDS/test/bicubic',
                        help='测试集低分辨率图像目录')
    parser.add_argument('--test_hr_dir', type=str,
                        default='./Datasets/VSR/REDS/test/GT',
                        help='测试集高分辨率图像目录')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='批大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--n_frames', type=int, default=5, help='输入帧数')
    parser.add_argument('--patch_size', type=int, default=256, help='训练patch大小')
    parser.add_argument('--scale', type=int, default=4, help='超分比例')

    # 模型参数
    parser.add_argument('--model', type=str, default='SimpleVSR',
                        choices=['SimpleVSR', 'BasicVSR'], help='模型选择')
    parser.add_argument('--num_features', type=int, default=64, help='特征通道数')
    parser.add_argument('--num_blocks', type=int, default=10, help='残差块数量')

    # 设备参数
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')

    # 实验管理
    parser.add_argument('--exp_name', type=str, default='vsr_exp', help='实验名称')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--result_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练检查点路径')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_freq', type=int, default=10, help='保存检查点频率')
    parser.add_argument('--val_freq', type=int, default=5, help='验证频率')
    parser.add_argument('--log_freq', type=int, default=20, help='日志打印频率')

    args = parser.parse_args()

    # 创建目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.exp_dir = os.path.join(args.checkpoint_dir, f"{args.exp_name}_{timestamp}")
    args.log_path = os.path.join(args.log_dir, f"{args.exp_name}_{timestamp}.log")
    args.result_dir = os.path.join(args.result_dir, f"{args.exp_name}_{timestamp}")

    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    return args


if __name__ == '__main__':
    config = get_config()
    print("配置参数:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")