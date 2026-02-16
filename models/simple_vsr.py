import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, num_features=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return out


class SimpleVSR(nn.Module):
    """简单的视频超分辨率模型"""

    def __init__(self, n_frames=5, num_features=64, num_blocks=10, scale=4):
        super(SimpleVSR, self).__init__()
        self.n_frames = n_frames
        self.scale = scale

        # 特征提取
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3 * n_frames, num_features, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 残差块
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )

        # 上采样
        if scale == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
        elif scale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError(f"不支持的比例因子: {scale}")

        # 重建层
        self.reconstruct = nn.Conv2d(num_features, 3, 3, padding=1)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) 输入低分辨率序列
        Returns:
            (B, C, H*scale, W*scale) 输出高分辨率图像
        """
        B, T, C, H, W = x.shape

        # 将时间维度与通道维度合并
        x = x.view(B, T * C, H, W)

        # 特征提取
        feat = self.feat_extract(x)

        # 残差块
        feat = self.residual_blocks(feat)

        # 上采样
        feat = self.upsample(feat)

        # 重建
        out = self.reconstruct(feat)

        return out