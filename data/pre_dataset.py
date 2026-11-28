import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, Subset
import numpy as np
from monai import data, transforms
import json
import nibabel as nib
import torchio as tio




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


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


class test_LA(Dataset):
    def __init__(self, data_path, data_txt, mode='train', transform=None):
        """
        增强版3D心脏MRI分割数据集类
        
        参数:
            data_path (str): 数据根目录
            data_txt (str): 包含要加载的子文件夹名称的文本文件路径
            mode (str): 数据集模式 ('train' 或 'test')
            transform (callable, optional): 数据增强转换函数
        """
        self.transform = transform
        self.paths = []
        self.mode = mode
        
        # 确定基础路径
        if mode == 'train':
            base_path = os.path.join(data_path, 'Training')
        elif mode == 'test':
            base_path = os.path.join(data_path, 'Testing')
        else:
            raise ValueError(f"无效的模式: {mode}. 必须是 'train' 或 'test'")
        
        # 检查base_path是否存在
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"无法找到基础目录: {base_path}")
        
        # 从文本文件中读取指定的子文件夹名称
        if not os.path.isfile(data_txt):
            raise FileNotFoundError(f"无法找到数据列表文件: {data_txt}")
            
        with open(data_txt, 'r') as f:
            case_names = [line.strip() for line in f.readlines() if line.strip()]
        
        if not case_names:
            warnings.warn(f"数据列表文件 {data_txt} 为空", RuntimeWarning)
        
        # 构建完整路径并验证存在性
        for case_name in case_names:
            case_path = os.path.join(base_path, case_name)
            if os.path.isdir(case_path):
                # 验证关键文件是否存在
                mri_path = os.path.join(case_path, 'mri.nii.gz')
                if os.path.isfile(mri_path):
                    self.paths.append(case_path)
                else:
                    warnings.warn(f"跳过 {case_path}: 缺少 mri.nii.gz 文件", RuntimeWarning)
            else:
                warnings.warn(f"跳过 {case_name}: 目录不存在", RuntimeWarning)
        
        if not self.paths:
            raise RuntimeError(f"在 {base_path} 下没有找到任何有效数据集")
    
    def __getitem__(self, item):
        """
        获取单个样本的数据和元数据
        
        返回:
            image (np.ndarray): 预处理后的MRI图像数据
            label (np.ndarray): 对应的标签掩码
            case_path (str): 当前样本的完整路径
        """
        case_path = self.paths[item]
        image_path = os.path.join(case_path, 'mri.nii.gz')
        label_path = os.path.join(case_path, 'endo.nii.gz')
        
        # 加载NIfTI图像
        try:
            nii_image = nib.load(image_path)
            image = nii_image.get_fdata().astype(np.float32)
            affine = nii_image.affine.copy()  # 保存几何信息
            
            # 只加载训练集的标签
            if self.mode == 'train':
                label_nii = nib.load(label_path)
                label = label_nii.get_fdata().astype(np.int16)
            else:
                label = np.zeros(image.shape, dtype=np.int16)  # 测试集返回空标签
        except Exception as e:
            raise IOError(f"加载 {case_path} 数据失败: {e}")
        
        # 添加通道维度 (1, D, H, W)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        # 对非背景数据进行归一化
        non_background_mask = image != 0
        if np.any(non_background_mask):
            non_background_data = image[non_background_mask]
            min_val = np.min(non_background_data)
            max_val = np.max(non_background_data)
            
            # 防止出现零除错误
            if max_val > min_val + 1e-5:
                image[non_background_mask] = (non_background_data - min_val) / (max_val - min_val)
        
        # 只处理训练集的标签
        if self.mode == 'train':
            if label.ndim == 3:
                label = np.expand_dims(label, axis=0)
            
            # 确保标签为整数类型
            label = label.astype(np.int16)
        else:
            # 对于验证/测试集，我们可以不返回标签，但为了统一接口，返回全零数组
            label = np.zeros(image.shape[:1] + label.shape[1:], dtype=np.int16)
        
        # 创建样本并应用转换
        sample = {
            'image': image,
            'label': label,
            #'case_path': case_path,
            #'affine': affine  # 保存几何信息
        }
        
        fisher = {'case_path': case_path,
            'affine': affine  }
        
        if self.transform:
            sample = self.transform(sample)
        
        # 返回图像、标签和路径
        return sample['image'], sample['label'], fisher['case_path']
    
    def __len__(self):
        return len(self.paths)
    
    def collate(self, batch):
        """
        自定义批处理函数
        
        处理包含图像、标签和路径的样本
        """
        images = []
        labels = []
        paths = []
        
        for b in batch:
            images.append(torch.as_tensor(b[0]))
            labels.append(torch.as_tensor(b[1]))
            paths.append(b[2])
        
        # 堆叠所有batch中的图像和标签
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        
        return images, labels, paths


class pre_LA(Dataset):
    def __init__(self, data_path, data_txt, mode='train', transform=None):
        """
        初始化数据集
        :param data_path: 数据根目录
        :param data_txt: 包含要加载的子文件夹名称的文本文件路径
        :param mode: 数据集模式 ('train' 或 'test')
        :param transform: 数据转换函数
        """
        self.transform = transform
        self.paths = []
        
        # 确定基础路径
        if mode == 'train':
            base_path = os.path.join(data_path, 'Training')
        elif mode == 'test':
            base_path = os.path.join(data_path, 'Testing')
        else:
            raise ValueError(f"无效的模式: {mode}. 必须是 'train' 或 'test'")
        
        # 从文本文件中读取指定的子文件夹名称
        try:
            with open(data_txt, 'r') as f:
                case_names = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"无法找到指定的数据列表文件: {data_txt}")
        
        # 构建完整路径
        for case_name in case_names:
            case_path = os.path.join(base_path, case_name)
            if os.path.isdir(case_path):
                self.paths.append(case_path)
            else:
                print(f"警告: 指定的子文件夹不存在: {case_path}")
    
    def __getitem__(self, item):
        # 构建图像和标签路径
        case_path = self.paths[item]
        image_path = os.path.join(case_path, 'mri.nii.gz')
        label_path = os.path.join(case_path, 'endo.nii.gz')
        
        # 加载NIfTI图像
        try:
            nii_image = nib.load(image_path)
            image = nii_image.get_fdata()
            label_nii = nib.load(label_path)
            label = label_nii.get_fdata()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"无法加载数据: {e}")
        
        # 添加通道维度
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        if label.ndim == 3:
            label = np.expand_dims(label, axis=0)
        
        # 对非背景数据进行归一化
        non_background_mask = image != 0
        if np.any(non_background_mask):
            non_background_data = image[non_background_mask]
            min_val = non_background_data.min()
            max_val = non_background_data.max()
            
            if max_val > min_val:  # 避免除以零
                image[non_background_mask] = (non_background_data - min_val) / (max_val - min_val)
        
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        
        return sample['image'], sample['label']
    
    def __len__(self):
        return len(self.paths)
    
    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]



class LA(Dataset):
    def __init__(self, data_path, mode= 'Training', transform=None, noise_rate=0.0, elastic_num = 10):
        
        if mode == 'train':
            file_path = os.path.join(data_path, 'Training')
            self.paths = []
            for case in os.listdir(file_path):
                if os.path.isdir(os.path.join(file_path, case)):
                    self.paths.append(os.path.join(file_path, case))
        
        elif mode == 'test':
            file_path = os.path.join(data_path, 'Testing')
            self.paths = []
            for case in os.listdir(file_path):
                if os.path.isdir(os.path.join(file_path, case)):
                    self.paths.append(os.path.join(file_path, case))        
               
        
        self.transform = transform
        self.noise_rate = noise_rate
        
        # 定义弹性变形参数
        self.transformation = tio.RandomElasticDeformation(
            num_control_points=elastic_num,  # 控制点数量（影响变形复杂度）# 之前都是10
            max_displacement=float(elastic_num),  # 最大位移量（单位：毫米）
            locked_borders=2,       # 边缘不移动
            image_interpolation='nearest'  # 保持二值标签
)

    def __getitem__(self, item):
        '''
        image_path = os.path.join(self.paths[item],'/image.nii.gz')
        label_path = os.path.join(self.paths[item],'/label.nii.gz')
        '''
        image_path  = self.paths[item] + '/mri.nii.gz'
        label_path = self.paths[item] + '/endo.nii.gz'

        nii_image = nib.load(image_path)
        image = nii_image.get_fdata()
        
        # 生成一个0到1之间的随机数
        random_number = random.random()
        #print(random_number)
        if random_number < self.noise_rate:
           #print(random_number)
           # 加载标签数据
           subject = tio.Subject(
               label=tio.LabelMap(label_path)
           )
           
           # 应用变形
           transformed_subject = self.transformation(subject)
           # 直接提取变形后的标签数据并转换为 NumPy 数组
           label_array = transformed_subject.label.data.numpy()  # 提取数据并转换为 NumPy 数组
           label = np.squeeze(label_array)  # 去除多余的维度（如果有）
        else:
           # 读取标签数据，这里假设标签也是 NIfTI 格式
           label_nii = nib.load(label_path)
           label = label_nii.get_fdata()
           # 将数据从 (x, y, z) 转换为 (D, H, W)
           
        
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        if label.ndim == 3:
            label = np.expand_dims(label, axis=0)   
            
        ## 对非背景数据进行归一化
        # 找到非背景数据
        data = image
        non_background_data = data[data != 0]
        # 计算非背景数据的最大和最小值
        non_background_min = non_background_data.min()
        non_background_max = non_background_data.max()

        # 归一化非背景数据到0-1范围
        # 注意：这里我们保持背景值为0
        non_background_indices = data != 0
        data[non_background_indices] = (data[non_background_indices] - non_background_min) / (non_background_max - non_background_min)
        image = data
        
        #print(data.shape)
        
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]



class BraTS_noise(Dataset):
    def __init__(self, file_list, transform=None, noise_rate=0.0):
        self.file_list = file_list
        self.transform = transform
        self.noise_rate = noise_rate

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        
        channels = []
        image_path = self.file_list[index]['image']
        num_channel = len(image_path)
        for channel in range(num_channel):
            # 读取NIfTI图像
            img = nib.load(image_path[channel])
            data = img.get_fdata()
            
            ## 对非背景数据进行归一化
            # 找到非背景数据
            non_background_data = data[data != 0]
            # 计算非背景数据的最大和最小值
            non_background_min = non_background_data.min()
            non_background_max = non_background_data.max()

            # 归一化非背景数据到0-1范围
            # 注意：这里我们保持背景值为0
            non_background_indices = data != 0
            data[non_background_indices] = (data[non_background_indices] - non_background_min) / (non_background_max - non_background_min)
            

            # 将数据从 (x, y, z) 转换为 (D, H, W)
            # data = np.transpose(data, (2, 1, 0))
            channels.append(torch.tensor(data, dtype=torch.float32))
        
        # 加载你的图像和标签数据
        image = np.stack(channels)
        
        #image = np.transpose(image, (0, 2, 3, 0))
        label_path = self.file_list[index]['label']
        
        # 生成一个0到1之间的随机数
        random_number = random.random()
        #print(random_number)
        if random_number < self.noise_rate:
        
           # 加载标签数据
           subject = tio.Subject(
               label=tio.LabelMap(label_path)
           )
           
           # 应用变形
           transformed_subject = transformation(subject)
           # 直接提取变形后的标签数据并转换为 NumPy 数组
           label_array = transformed_subject.label.data.numpy()  # 提取数据并转换为 NumPy 数组
           label = np.squeeze(label_array)  # 去除多余的维度（如果有）
        else:
           # 读取标签数据，这里假设标签也是 NIfTI 格式
           label_nii = nib.load(label_path)
           label = label_nii.get_fdata()
           # 将数据从 (x, y, z) 转换为 (D, H, W)
        
        ## 转成独热编码
        # 获取标签数据的形状
        height, width, depth = label.shape
        
        # 创建独热编码，形状为 (4, D, H, W)
        num_classes = 3  # 4 个通道：0, 1, 2, 1+2
        label_onehot = np.zeros((num_classes, height, width, depth), dtype=np.uint8)
        
        label_onehot[0] = np.logical_or(label == 1, label == 4).astype(np.uint8)
        label_onehot[1] = np.logical_or(label == 1, label == 4, label == 2).astype(np.uint8)
        label_onehot[2] = (label == 4).astype(np.uint8)
        
        '''
        # 填充前 3 个通道（独热编码）
        for i in range(3):  # 0, 1, 2
            label_onehot[i] = (label_data == i).astype(np.uint8)
        
        # 填充第四个通道（标签 1 和标签 2 的和）
        label_onehot[3] = np.logical_or(label_data == 1, label_data == 2).astype(np.uint8)
        '''
        
        
        #label = torch.tensor(label_nii.get_fdata().astype(int), dtype=torch.long)
        #label[label == 4] = 3
        
        # 如果有变换，应用变换
        if self.transform:
            sample = self.transform({'image': image, 'label': label_onehot})
            image = sample['image']
            label = sample['label']
           
        return (image,label)
    
    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
        
        


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
        


    if args.test_mode:
        #val_ds = data.Dataset(data=validation_files, transform=test_transform)
        #val_ds = LA(data_path, mode= 'testing', transform=test_transform, noise_rate=0.0)
        val_ds = test_LA(data_path=data_dir, data_txt=args.val_txt , mode= 'train', transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )

        loader = test_loader
    else:
        train_ds = pre_LA(data_path=data_dir, data_txt=args.train_txt , mode= 'train', transform=train_transform)
        
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
        val_ds = pre_LA(data_path=data_dir, data_txt=args.val_txt , mode= 'train', transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader




class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (c, w, h, d) = image.shape
        
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[:,w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[:,w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
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





