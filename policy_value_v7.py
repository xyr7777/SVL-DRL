import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]
        
        # 应用合适的填充
        pad = [
            diffZ // 2, diffZ - diffZ // 2,
            diffY // 2, diffY - diffY // 2,
            diffX // 2, diffX - diffX // 2
        ]
        x1 = F.pad(x1, pad)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PolicyValueSegmentationNetwork(nn.Module):
    def __init__(self, in_channels=1, seg_channels=3, bilinear=False):
        super(PolicyValueSegmentationNetwork, self).__init__()
        self.bilinear = bilinear

        # 编码器部分
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        
        # 瓶颈层
        self.bottleneck = DoubleConv(256, 256)
        
        # 分割解码器部分
        self.up1_seg = Up(256, 128, bilinear)
        self.up2_seg = Up(128, 64, bilinear)
        self.up3_seg = Up(64, 32, bilinear)
        self.up4_seg = Up(32, 16, bilinear)
        self.seg_out = OutConv(16, seg_channels)
        
        # 策略解码器 - 将输出调整为3D概率图[B, C, D, H, W]
        self.policy_decoder = self._create_policy_decoder()
        
        # 价值解码器
        self.value_decoder = self._create_value_decoder()
        
        # 存储编码器特征
        self.encoding_features = []
        
        # 策略解码器的上采样模块
        self.policy_up1 = Up(256, 128, bilinear)
        self.policy_up2 = Up(128, 64, bilinear)
        self.policy_up3 = Up(64, 32, bilinear)
        self.policy_up4 = Up(32, 16, bilinear)
        self.policy_out = OutConv(16, 4)  # 输出4个动作类别的概率图

    def forward(self, x):
        input_size = x.shape[2:]  # 保存输入尺寸用于策略输出
        
        # 编码器部分
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 存储特征供策略和价值解码器使用
        self.encoding_features = [x1, x2, x3, x4]  # x1是最浅层特征
        
        # 瓶颈层
        bottleneck = self.bottleneck(x5)
        
        # 分割解码器
        seg = self.up1_seg(bottleneck, x4)
        seg = self.up2_seg(seg, x3)
        seg = self.up3_seg(seg, x2)
        seg = self.up4_seg(seg, x1)
        seg = self.seg_out(seg)
        
        # 策略解码器 - 输出3D概率图
        policy = self.forward_policy(bottleneck, input_size)
        
        # 价值解码器
        value = self.forward_value(bottleneck)
        
        return seg, policy, value
    
    def _create_policy_decoder(self):
        """策略解码器：使用卷积层处理特征"""
        return nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
    
    def _create_value_decoder(self):
        """价值解码器：修正特征维度"""
        return nn.Sequential(
            nn.Linear(256 + 128 + 64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
    
    def forward_policy(self, bottleneck, input_size):
        """策略解码器前向传播 - 输出3D概率图"""
        # 使用策略解码器的特征处理
        x = self.policy_decoder(bottleneck)
        
        # 逐步上采样到原始输入尺寸
        x = self.policy_up1(x, self.encoding_features[3])
        x = self.policy_up2(x, self.encoding_features[2])
        x = self.policy_up3(x, self.encoding_features[1])
        x = self.policy_up4(x, self.encoding_features[0])
        
        # 输出4个动作类别的概率图
        policy = self.policy_out(x)
        
        # 上采样到原始输入尺寸（如果尺寸不完全匹配）
        if policy.shape[2:] != input_size:
            policy = F.interpolate(policy, size=input_size, mode='trilinear', align_corners=True)
        
        # 沿通道维度应用softmax，确保每个空间位置的通道概率和为1
        policy = F.softmax(policy, dim=1)
        
        return policy
    
    def forward_value(self, bottleneck):
        """价值解码器前向传播"""
        # 获取所有特征图
        x1, x2, x3, x4 = self.encoding_features
        
        # 对所有特征进行全局平均池化
        feat_x1 = F.adaptive_avg_pool3d(x1, 1).view(x1.size(0), -1)
        feat_x2 = F.adaptive_avg_pool3d(x2, 1).view(x2.size(0), -1)
        feat_x3 = F.adaptive_avg_pool3d(x3, 1).view(x3.size(0), -1)
        feat_x4 = F.adaptive_avg_pool3d(x4, 1).view(x4.size(0), -1)
        feat_bottleneck = F.adaptive_avg_pool3d(bottleneck, 1).view(bottleneck.size(0), -1)
        
        # 合并特征
        combined = torch.cat([feat_bottleneck, feat_x4, feat_x3, feat_x2], dim=1)
        
        # 通过价值解码器
        value = self.value_decoder(combined)
        return value
    
    # 辅助方法用于分阶段训练
    def forward_segmentation_only(self, x):
        """仅用于分割任务的前向传播"""
        # 编码器部分
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 瓶颈层
        bottleneck = self.bottleneck(x5)
        
        # 分割解码器
        seg = self.up1_seg(bottleneck, x4)
        seg = self.up2_seg(seg, x3)
        seg = self.up3_seg(seg, x2)
        seg = self.up4_seg(seg, x1)
        seg = self.seg_out(seg)
        
        return seg