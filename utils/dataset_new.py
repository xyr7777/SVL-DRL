import h5py
import numpy as np
import os
import random
from torch.utils.data import Dataset
import torch
from monai import data, transforms
import SimpleITK as sitk



def random_morphology_3d(label_array, label_values, max_kernel_size=3):
    """
    对多标签3D标签进行随机腐蚀/扩张
    Args:
        label_array: 3D numpy数组，包含多个标签值
        label_values: 列表，需要操作的标签值（如[1,2,3]）
        max_kernel_size: 最大形态学核大小（奇数）
    Returns:
        处理后的标签数组
    """   

    # 随机选择腐蚀或扩张
    operation = random.choice(['erode', 'dilate'])
    kernel_size = random.randint(1, max_kernel_size)
    
    # 创建球形核（3D）
    kernel = sitk.Ball(kernel_size)
    
    result_array = np.zeros_like(label_array)
    for label in label_values:
        # 提取当前标签的二进制掩码
        binary_mask = (label_array == label).astype(np.uint8)
        
        if np.sum(binary_mask) == 0:
            continue  # 跳过不存在的标签
            
        # 转换为SimpleITK图像
        sitk_mask = sitk.GetImageFromArray(binary_mask)
        
        # 执行腐蚀或扩张
        if operation == 'erode':
            processed = sitk.BinaryErode(sitk_mask, kernel)
        else:
            processed = sitk.BinaryDilate(sitk_mask, kernel)
        
        # 转换回numpy并合并结果
        processed_array = sitk.GetArrayFromImage(processed)
        result_array[processed_array > 0] = label
    
    # 保留未操作区域的原始标签
    result_array[result_array == 0] = label_array[result_array == 0]
    return result_array



def get_loader(args):
    data_dir = args.data_dir
    
    train_transform = transforms.Compose([
            #RandomRotFlip(),
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
    '''
    if args.data_name == 'Brats21':
        datalist_json = args.json_list
        train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold)
    '''    


    if args.test_mode:
        #val_ds = data.Dataset(data=validation_files, transform=test_transform)
        val_ds = LA(data_path, mode= 'testing', transform=test_transform, noise_rate=0.0)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )

        loader = test_loader
    else:
        if args.data_name == 'left_atrial':
            train_ds = LA(data_path=data_dir, mode= 'train', transform=train_transform, noise_rate=args.noise_rate, elastic_num = args.elastic_num)
        elif args.data_name == 'Pancreas':
            train_ds = Pancreas(base_dir=data_dir, split= 'train', transform=train_transform, noise_rate=args.noise_rate, elastic_num = args.elastic_num)
        elif args.data_name == 'Brats21':
            #train_ds = BraTS_noise(file_list=train_files,transform=train_transform, noise_rate=args.noise_rate, elastic_num = args.elastic_num)
            train_ds = Brats(base_dir=data_dir,transform=train_transform, noisy_ratio=args.noise_rate)
            #base_dir=None, split='train', num=None, transform=None, noisy_ratio=0.5
        elif args.data_name == 'SFDA_LA': 
            train_ds = SFDA_LA(data_path=data_dir, noise_label_path=args.noise_label_path, data_txt=args.noise_label_txt, mode='train', transform=train_transform)
        elif args.data_name == 'source_LA': 
            train_ds = MSD_LA(json_path = "/home/shiteam_public2/seg/data/Task02_Heart/dataset.json", root_dir = data_dir, mode='train', transform=train_transform)
        elif args.data_name == 'source_Pancreas': 
            train_ds = MSD_Pancreas(json_path = "/home/shiteam_public2/seg/data/Task07_Pancreas/dataset.json", root_dir = data_dir, mode='train', transform=train_transform)
        elif args.data_name == 'MT_Pancreas': 
            train_ds = MT_Pancreas(base_dir=data_dir, split='train', num=None, transform=train_transform, noisy_ratio=args.noise_rate)
        elif args.data_name == 'MT_LA': 
            train_ds = MT_LA(base_dir=data_dir, split='train', num=None, transform=train_transform, noisy_ratio=args.noise_rate)
        
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        #val_ds = data.Dataset(data=validation_files, transform=val_transform)
        if args.data_name == 'left_atrial':
            val_ds = LA(data_path=data_dir, mode= 'test', transform=val_transform, noise_rate=0.0, elastic_num = args.elastic_num)
        elif args.data_name == 'Pancreas':
            val_ds = Pancreas(base_dir=data_dir, split= 'test', transform=val_transform, noise_rate=0.0, elastic_num = args.elastic_num) 
        elif args.data_name == 'Brats21':
            #val_ds = BraTS_noise(file_list=validation_files, transform=val_transform, noise_rate=0.0, elastic_num = args.elastic_num)
            val_ds = Brats(base_dir=data_dir,transform=val_transform, noisy_ratio=0.0)   
        elif args.data_name == 'SFDA_LA':    
            val_ds = SFDA_LA(data_path=data_dir, noise_label_path=args.noise_label_path, data_txt=args.noise_label_txt, mode='test', transform=val_transform)
        elif args.data_name == 'source_LA': 
            val_ds = LA_heart(base_dir='/home/shiteam_public2/seg/data/LASeg/', split='train', num=None, transform=val_transform)
            
        elif args.data_name == 'source_Pancreas': 
            val_ds = Pancreas(base_dir="/home/shiteam_public2/seg/data/Pancreas/", split= 'train', transform=val_transform, noise_rate=0.0, elastic_num = args.elastic_num) 
        
        elif args.data_name == 'MT_Pancreas': 
            val_ds = MT_Pancreas(base_dir=data_dir, split='test', num=None, transform=val_transform, noisy_ratio=0.0)
            
        elif args.data_name == 'MT_LA': 
            val_ds = MT_LA(base_dir=data_dir, split='test', num=None, transform=val_transform, noisy_ratio=0.0)
            
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader



class Brats(Dataset):
    """ Pancreas Dataset with fixed ratio of noisy/original labels """
    
    def __init__(self, base_dir=None, split='train', num=None, transform=None, noisy_ratio=0.5):
        """
        Args:
            base_dir (str): 数据集根目录
            split (str): 'train' 或 'test'
            num (int, optional): 使用的样本数量
            transform (callable, optional): 数据转换函数
            noisy_ratio (float): 使用noisy_label的比例 (0.0 - 1.0)
        """
        self._base_dir = base_dir
        self.transform = transform
        self.split = split
        self.noisy_ratio = noisy_ratio
        
        # 读取文件列表
        train_path = "/home/shiteam_public2/seg/data/brats-2021-task1/train.txt"
        test_path = "/home/shiteam_public2/seg/data/brats-2021-task1/test.txt"

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = [line.strip() for line in f.readlines()]
            
            # 按比例划分噪声标签和原始标签
            total_samples = len(self.image_list)
            self.noisy_split = int(total_samples * noisy_ratio)
            print(f"Train set: Using {self.noisy_split}/{total_samples} samples with noisy labels")
            
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = [line.strip() for line in f.readlines()]
            print("Test set: Using original labels only")

        if num is not None:
            self.image_list = self.image_list[:num]
            
        print(f"Total {len(self.image_list)} {split} samples | Noisy ratio: {noisy_ratio}")

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5_path = os.path.join(self._base_dir, "H5Data", f"{image_name}.h5")
        
        with h5py.File(h5_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]
            
            # 训练集：前noisy_split个用噪声处理，其余用原始标签
            if self.split == 'train' and idx < self.noisy_split:
                label = random_morphology_3d(label, (1,2,4), max_kernel_size=3)
            
        image = np.moveaxis(image, -1, 0)
        # 确保数据是三维格式 (C, D, H, W)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0) 
        if label.ndim == 3:
            # 初始化三个通道
            et = (label == 4).astype(np.uint8)          # 增强肿瘤 (ET)
            tc = np.logical_or(label == 1, label == 4).astype(np.uint8)  # 肿瘤核心 (TC)
            wt = np.logical_or(np.logical_or(label == 1, label == 2), label == 4).astype(np.uint8)  # 整个肿瘤 (WT)
            
            # 堆叠成多通道
            label = np.stack([et, wt, tc], axis=0)  # shape (3, D, H, W)
        
        
        # 创建样本字典并应用转换
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        
        return sample['image'], sample['label']





class MT_Pancreas(Dataset):
    """ Pancreas Dataset with fixed ratio of noisy/original labels """
    
    def __init__(self, base_dir=None, split='train', num=None, transform=None, noisy_ratio=0.5):
        """
        Args:
            base_dir (str): 数据集根目录
            split (str): 'train' 或 'test'
            num (int, optional): 使用的样本数量
            transform (callable, optional): 数据转换函数
            noisy_ratio (float): 使用noisy_label的比例 (0.0 - 1.0)
        """
        self._base_dir = base_dir
        self.transform = transform
        self.split = split
        self.noisy_ratio = noisy_ratio
        
        # 读取文件列表
        train_path = os.path.join(base_dir, 'train.list')
        test_path = os.path.join(base_dir, 'test.list')

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = [line.strip() for line in f.readlines()]
            
            # 按比例划分噪声标签和原始标签
            total_samples = len(self.image_list)
            self.noisy_split = int(total_samples * noisy_ratio)
            print(f"Train set: Using {self.noisy_split}/{total_samples} samples with noisy labels")
            
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = [line.strip() for line in f.readlines()]
            print("Test set: Using original labels only")

        if num is not None:
            self.image_list = self.image_list[:num]
            
        print(f"Total {len(self.image_list)} {split} samples | Noisy ratio: {noisy_ratio}")

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5_path = os.path.join(self._base_dir, "noisy_data", f"noisy_{image_name}.h5")
        
        with h5py.File(h5_path, 'r') as h5f:
            image = h5f['image'][:]
            
            # 训练集：前noisy_split个用噪声标签，其余用原始标签
            if self.split == 'train':
                label = h5f['noisy_label'][:] if idx < self.noisy_split else h5f['original_label'][:]
            # 测试集：始终用原始标签
            else:
                label = h5f['original_label'][:]
        
        # 确保数据是三维格式 (C, D, H, W)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0) 
        if label.ndim == 3:
            label = np.expand_dims(label, axis=0)
        
        # 创建样本字典并应用转换
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        
        return sample['image'], sample['label']
        
      
      
class MT_LA(Dataset):
    """ LA Dataset with fixed ratio of noisy/original labels """
    
    def __init__(self, base_dir=None, split='train', num=None, transform=None, noisy_ratio=0.5):
        """
        Args:
            base_dir (str): 数据集根目录
            split (str): 'train' 或 'test'
            num (int, optional): 使用的样本数量
            transform (callable, optional): 数据转换函数
            noisy_ratio (float): 使用noisy_label的比例 (0.0 - 1.0)
        """
        self._base_dir = base_dir
        self.transform = transform
        self.split = split
        self.noisy_ratio = noisy_ratio
        
        # 读取文件列表
        train_path = os.path.join(base_dir, 'train.list')
        test_path = os.path.join(base_dir, 'test.list')

        if split == 'train':
            with open(train_path, 'r') as f:
                self.case_list = [line.strip() for line in f.readlines()]
            
            # 按比例划分噪声标签和原始标签
            total_samples = len(self.case_list)
            self.noisy_split = int(total_samples * noisy_ratio)
            print(f"Train set: Using {self.noisy_split}/{total_samples} cases with noisy labels")
            
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.case_list = [line.strip() for line in f.readlines()]
            print("Test set: Using original labels only")

        if num is not None:
            self.case_list = self.case_list[:num]
            
        print(f"Total {len(self.case_list)} {split} cases | Noisy ratio: {noisy_ratio}")

    def __len__(self):
        return len(self.case_list)
    
    def __getitem__(self, idx):
        case_id = self.case_list[idx]
        h5_path = os.path.join(self._base_dir, 'SFDA_noisy_data', case_id, "mri_norm2_noisy.h5")
        
        with h5py.File(h5_path, 'r') as h5f:
            image = h5f['image'][:]
            
            # 训练集：前noisy_split个用噪声标签，其余用原始标签
            if self.split == 'train':
                label = h5f['noisy_label'][:] if idx < self.noisy_split else h5f['original_label'][:]
            # 测试集：始终用原始标签
            else:
                label = h5f['original_label'][:]
        
        # 确保数据是三维格式 (C, D, H, W)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0) 
        if label.ndim == 3:
            label = np.expand_dims(label, axis=0)
        
        # 创建样本字典并应用转换
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        
        return sample['image'], sample['label']      
      
      
      
      
      
        
class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


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
