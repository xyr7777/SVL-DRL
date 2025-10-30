# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)



class AverageMeter(object):
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
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out

# 自定义 Jaccard Index (IoU) 计算函数
def compute_jaccard(y_pred, y_true):
    """
    计算 Jaccard Index (IoU)
    Args:
        y_pred: 预测的分割结果 (二值图像)
        y_true: 真实的分割标签 (二值图像)
    Returns:
        jaccard: Jaccard Index 值
    """
    intersection = np.logical_and(y_pred, y_true).sum()
    union = np.logical_or(y_pred, y_true).sum()
    
    if union == 0:
        return 1.0  # 如果两者都为空，定义为完美匹配
    return intersection / union



def compute_hd95(y_pred, y_true, voxelspacing=None, sample_rate=0.1):
    """
    高效计算95% Hausdorff Distance - 使用KD树和采样
    """
    # 提取表面点
    pred_surface = np.where(ndimage.binary_erosion(y_pred) != y_pred)
    true_surface = np.where(ndimage.binary_erosion(y_true) != y_true)
    
    if len(pred_surface[0]) == 0 or len(true_surface[0]) == 0:
        return 0.0
    
    # 转换为点坐标
    pred_points = np.array(pred_surface).T
    true_points = np.array(true_surface).T
    
    # 对点集进行下采样以减少计算量
    if len(pred_points) > 5000:
        indices = np.random.choice(len(pred_points), 
                                  size=min(5000, int(len(pred_points)*sample_rate)), 
                                  replace=False)
        pred_points = pred_points[indices]
    
    if len(true_points) > 5000:
        indices = np.random.choice(len(true_points), 
                                  size=min(5000, int(len(true_points)*sample_rate)), 
                                  replace=False)
        true_points = true_points[indices]
    
    # 考虑体素间距
    if voxelspacing is not None:
        pred_points = pred_points * np.array(voxelspacing)
        true_points = true_points * np.array(voxelspacing)
    
    # 使用KD树高效计算距离
    tree_true = cKDTree(true_points)
    d_pred_to_true, _ = tree_true.query(pred_points, k=1)
    
    tree_pred = cKDTree(pred_points)
    d_true_to_pred, _ = tree_pred.query(true_points, k=1)
    
    # 取95百分位数
    hd95_1 = np.percentile(d_pred_to_true, 95)
    hd95_2 = np.percentile(d_true_to_pred, 95)
    
    return max(hd95_1, hd95_2)

def compute_asd(y_pred, y_true, voxelspacing=None, sample_rate=0.1):
    """
    高效计算Average Surface Distance - 使用KD树和采样
    """
    # 提取表面点
    pred_surface = np.where(ndimage.binary_erosion(y_pred) != y_pred)
    true_surface = np.where(ndimage.binary_erosion(y_true) != y_true)
    
    if len(pred_surface[0]) == 0 or len(true_surface[0]) == 0:
        return 0.0
    
    # 转换为点坐标
    pred_points = np.array(pred_surface).T
    true_points = np.array(true_surface).T
    
    # 对点集进行下采样以减少计算量
    if len(pred_points) > 5000:
        indices = np.random.choice(len(pred_points), 
                                  size=min(5000, int(len(pred_points)*sample_rate)), 
                                  replace=False)
        pred_points = pred_points[indices]
    
    if len(true_points) > 5000:
        indices = np.random.choice(len(true_points), 
                                  size=min(5000, int(len(true_points)*sample_rate)), 
                                  replace=False)
        true_points = true_points[indices]
    
    # 考虑体素间距
    if voxelspacing is not None:
        pred_points = pred_points * np.array(voxelspacing)
        true_points = true_points * np.array(voxelspacing)
    
    # 使用KD树高效计算距离
    tree_true = cKDTree(true_points)
    d_pred_to_true, _ = tree_true.query(pred_points, k=1)
    
    tree_pred = cKDTree(pred_points)
    d_true_to_pred, _ = tree_pred.query(true_points, k=1)
    
    # 计算平均值
    asd_value = (np.mean(d_pred_to_true) + np.mean(d_true_to_pred)) / 2
    
    return asd_value




# 自定义指标计算器类
class CustomMetricsCalculator:
    def __init__(self, include_background=False, voxelspacing=None):
        self.include_background = include_background
        self.voxelspacing = voxelspacing
        self.dice_values = []
        self.jaccard_values = []
        self.hd95_values = []
        self.asd_values = []
        
    def reset(self):
        self.dice_values = []
        self.jaccard_values = []
        self.hd95_values = []
        self.asd_values = []
        
    def __call__(self, y_pred, y_true):
        # 确保数据是numpy数组
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
            
        # 确保是二值图像
        y_pred = (y_pred > 0.5).astype(np.uint8)
        y_true = (y_true > 0.5).astype(np.uint8)
        
        # 计算指标
        dice = self._compute_dice(y_pred, y_true)
        jaccard = compute_jaccard(y_pred, y_true)
        hd95 = compute_hd95(y_pred, y_true, self.voxelspacing)
        asd = compute_asd(y_pred, y_true, self.voxelspacing)
        
        # 存储结果
        self.dice_values.append(dice)
        self.jaccard_values.append(jaccard)
        self.hd95_values.append(hd95)
        self.asd_values.append(asd)
        
        return dice, jaccard, hd95, asd
    
    def _compute_dice(self, y_pred, y_true):
        """计算Dice系数"""
        intersection = np.logical_and(y_pred, y_true).sum()
        union = y_pred.sum() + y_true.sum()
        
        if union == 0:
            return 1.0
        return 2 * intersection / union
    
    def aggregate(self):
        """聚合所有计算结果"""
        dice_avg = np.mean(self.dice_values) if self.dice_values else 0.0
        jaccard_avg = np.mean(self.jaccard_values) if self.jaccard_values else 0.0
        hd95_avg = np.mean(self.hd95_values) if self.hd95_values else 0.0
        asd_avg = np.mean(self.asd_values) if self.asd_values else 0.0
        
        return dice_avg, jaccard_avg, hd95_avg, asd_avg