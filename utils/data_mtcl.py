import os
import random
import time
import warnings
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchio as tio
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

from monai.data import decollate_batch
from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple, Optional, Any
import argparse
from monai import data, transforms



class MTCLDataset(Dataset):
    """MTCL专用数据集类 - 支持高质量和低质量数据分离"""
    
    def __init__(self, data_root: str, mode: str = 'train', 
                 hq_cases: List[str] = None, is_hq_set: bool = False,
                 noise_rate: float = 0.5, transform: callable = None,
                 crop_size: Tuple[int, int, int] = (96, 96, 96),
                 elastic_deform: bool = True):
        """
        初始化MTCL数据集
        
        参数:
            data_root: 数据根目录
            mode: 模式 ('train', 'val', 'test')
            hq_cases: 高质量病例列表
            is_hq_set: 是否为高质量数据集
            noise_rate: 噪声率（低质量数据）
            transform: 数据变换
            crop_size: 裁剪尺寸
            elastic_deform: 是否启用弹性变形
        """
        self.mode = mode
        self.hq_cases = hq_cases or []
        self.is_hq_set = is_hq_set
        self.noise_rate = noise_rate
        self.transform = transform
        self.crop_size = crop_size
        self.elastic_deform = elastic_deform
        
        # 确定数据路径
        self.data_path = os.path.join(data_root, 'Training' if mode == 'train' else 'Testing')
        if not os.path.exists(self.data_path):
            raise ValueError(f"数据路径不存在: {self.data_path}")
        
        # 收集病例
        self.cases = self._collect_cases()
        if not self.cases:
            raise ValueError(f"在 {self.data_path} 中未找到有效病例")
        
        # 弹性变形变换
        if elastic_deform:
            self.deform_transform = tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=7.0,
                locked_borders=2
            )
        
        print(f"初始化MTCL数据集: 模式={mode}, 病例数={len(self.cases)}, "
              f"高质量={is_hq_set}, 噪声率={noise_rate}")

    def _collect_cases(self) -> List[Dict[str, str]]:
        """收集病例信息"""
        cases = []
        for case_name in os.listdir(self.data_path):
            case_dir = os.path.join(self.data_path, case_name)
            if os.path.isdir(case_dir):
                image_path = os.path.join(case_dir, 'mri.nii.gz')
                label_path = os.path.join(case_dir, 'endo.nii.gz')
                
                if all(os.path.exists(p) for p in [image_path, label_path]):
                    is_hq = case_name in self.hq_cases
                    cases.append({
                        'name': case_name,
                        'image_path': image_path,
                        'label_path': label_path,
                        'is_hq': is_hq
                    })
        return cases

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        case = self.cases[idx]
        
        try:
            # 加载图像和标签
            image = self._load_nifti(case['image_path'])
            label = self._load_nifti(case['label_path'])
            
            # 应用噪声（如果是低质量训练集）
            if self.mode == 'train' and not self.is_hq_set:
                label = self._apply_label_noise(label)
            
            # 确保数据是三维格式 (C, D, H, W)
            if image.ndim == 3:
                image = np.expand_dims(image, axis=0) 
            if label.ndim == 3:
                label = np.expand_dims(label, axis=0)
            
            sample = {
                'image': image,
                'label': label,
            }
            
            # 应用数据增强
            if self.transform:
                sample = self.transform(sample)
                
            return sample
            
        except Exception as e:
            raise RuntimeError(f"加载病例 {case['name']} 失败: {str(e)}")

    def _load_nifti(self, path: str) -> np.ndarray:
        """加载NIfTI文件"""
        nii = nib.load(path)
        data = nii.get_fdata().astype(np.float32)
        return data

    def _apply_label_noise(self, label: np.ndarray) -> np.ndarray:
        """应用标签噪声（弹性变形）"""
        if self.elastic_deform:
            # 创建torchio主题
            subject = tio.Subject(
                label=tio.ScalarImage(tensor=label[np.newaxis, ...])
            )
            # 应用变形
            transformed = self.deform_transform(subject)
            noisy_label = transformed.label.numpy()[0]  # 移除通道维度
            return noisy_label
        return label

    


def get_mtcl_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建MTCL数据加载器"""
    # 获取所有训练病例
    train_path = os.path.join(args.data_dir, 'Training')
    all_cases = [c for c in os.listdir(train_path) 
                if os.path.isdir(os.path.join(train_path, c))]
    
    # 选择高质量病例
    num_hq = max(1, int(len(all_cases) * args.hq_ratio))
    hq_cases = random.sample(all_cases, num_hq)
    
    train_transform = transforms.Compose([
            RandomRotFlip(),
            RandomCrop([args.roi_x, args.roi_y, args.roi_z]),
            #GaussianNoise(p=0.1),
            ToTensor()
        ])
    val_transform = transforms.Compose([
            #CenterCrop([args.roi_x, args.roi_y, args.roi_z]),
            ToTensor()
        ])
    
    test_transform = transforms.Compose([
            #CenterCrop([args.roi_x, args.roi_y, args.roi_z]),
            ToTensor()
        ])
    
    # 创建数据集
    hq_dataset = MTCLDataset(
        data_root=args.data_dir,
        mode='train',
        hq_cases=hq_cases,
        is_hq_set=True,
        noise_rate=0.0,  # 高质量数据无噪声
        transform=train_transform,
        crop_size=(args.roi_x, args.roi_y, args.roi_z),
        elastic_deform=False
    )
    
    lq_dataset = MTCLDataset(
        data_root=args.data_dir,
        mode='train',
        hq_cases=hq_cases,
        is_hq_set=False,
        noise_rate=args.noise_rate,
        transform=train_transform,
        crop_size=(args.roi_x, args.roi_y, args.roi_z),
        elastic_deform=True
    )
    
    val_dataset = MTCLDataset(
        data_root=args.data_dir,
        mode='val',
        hq_cases=[],
        is_hq_set=False,
        noise_rate=0.0,
        transform=val_transform,
        crop_size=(args.roi_x, args.roi_y, args.roi_z),
        elastic_deform=False
    )
    
    # 创建数据加载器
    hq_loader = DataLoader(
        hq_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    lq_loader = DataLoader(
        lq_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 验证通常使用batch_size=1
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    return hq_loader, lq_loader, val_loader, hq_cases
    
    
class RandomCrop(object):
    """
    随机裁剪样本中的图像，如果维度不足则进行填充
    
    参数:
    output_size (tuple): 期望的输出尺寸 (depth, height, width)
    padding_mode (str): 填充模式 ('constant', 'edge', 'reflect'等)
    padding_value (float): 填充值（对于图像）
    padding_label (int): 标签填充值（通常为背景值）
    """
    def __init__(self, output_size, padding_mode='constant', padding_value=0, padding_label=0):
        self.output_size = output_size
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.padding_label = padding_label

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        
        # 获取当前尺寸 (c, w, h, d)
        c, w, h, d = image.shape
        
        # 检查每个维度是否需要填充
        pad_width = []
        for i, dim in enumerate([w, h, d]):
            if dim < self.output_size[i]:
                # 计算需要填充的总量
                total_pad = self.output_size[i] - dim
                # 对称填充（前一半，后一半）
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_width.append((pad_before, pad_after))
            else:
                pad_width.append((0, 0))
        
        # 如果有任一维度需要填充
        if any(pad != (0, 0) for pad in pad_width):
            # 图像填充：在空间维度上填充，通道维度不填充
            image_padding = [(0, 0)]  # 通道维度不填充
            image_padding.extend(pad_width)
            image = np.pad(image, image_padding, 
                          mode=self.padding_mode, 
                          constant_values=self.padding_value)
            
            # 标签填充：在空间维度上填充
            label_padding = [(0, 0)]  # 如果有通道维度则保留
            label_padding.extend(pad_width)
            label = np.pad(label, label_padding, 
                          mode=self.padding_mode, 
                          constant_values=self.padding_label)
        
        # 更新填充后的尺寸
        _, w, h, d = image.shape
        
        # 计算随机裁剪的起始点
        w1 = np.random.randint(0, max(1, w - self.output_size[0]))
        h1 = np.random.randint(0, max(1, h - self.output_size[1]))
        d1 = np.random.randint(0, max(1, d - self.output_size[2]))
        
        # 执行裁剪
        image = image[:, 
                     w1:w1 + self.output_size[0], 
                     h1:h1 + self.output_size[1], 
                     d1:d1 + self.output_size[2]]
        
        label = label[:, 
                     w1:w1 + self.output_size[0], 
                     h1:h1 + self.output_size[1], 
                     d1:d1 + self.output_size[2]]
        
        return {'image': image, 'label': label}



class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (c, w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[:,w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[:,w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        

        return {'image': image, 'label': label}
        
class RandomRotFlip(object):
    """
    Randomly rotate and flip the 3D volume
    """

    def __call__(self, sample):
        if isinstance(sample, dict):
            image, label = sample['image'], sample['label']
        else:
            image, label = sample
        
        # 确保数据格式正确
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        if label.ndim == 3:
            label = np.expand_dims(label, axis=0)
        
        
        # 随机旋转
        k = np.random.randint(0, 4)
        # 对每个通道进行旋转
        rotated_image = np.stack([np.rot90(x, k) for x in image], axis=0)
        rotated_label = np.stack([np.rot90(x, k) for x in label], axis=0)
        #rotated_label = np.rot90(label, k)
        
        # 随机翻转
        axis = np.random.randint(1, 4)
        flipped_image = np.flip(rotated_image, axis=axis).copy()
        flipped_label = np.flip(rotated_label, axis=axis).copy()

        # 返回与输入相同的格式
        if isinstance(sample, dict):
            return {'image': flipped_image, 'label': flipped_label}
        else:
            return flipped_image, flipped_label
