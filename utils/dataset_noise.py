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
#from utils.dataset import Pancreas

import h5py
import itertools
from scipy import ndimage
import random
#from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
#from utils.dataset import *


def get_loader(args):
    data_dir = args.data_dir
    
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
    
    if args.data_name == 'Brats21':
        datalist_json = args.json_list
        train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold)    


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
            train_ds = BraTS_noise(file_list=train_files,transform=train_transform, noise_rate=args.noise_rate, elastic_num = args.elastic_num)
        elif args.data_name == 'SFDA_LA': 
            train_ds = SFDA_LA(data_path=data_dir, noise_label_path=args.noise_label_path, data_txt=args.noise_label_txt, mode='train', transform=train_transform)
        elif args.data_name == 'source_LA': 
            train_ds = MSD_LA(json_path = "/home/shiteam_public2/seg/data/Task02_Heart/dataset.json", root_dir = data_dir, mode='train', transform=train_transform)
        elif args.data_name == 'source_Pancreas': 
            train_ds = MSD_Pancreas(json_path = "/home/shiteam_public2/seg/data/Task07_Pancreas/dataset.json", root_dir = data_dir, mode='train', transform=train_transform)
        
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
            val_ds = BraTS_noise(file_list=validation_files, transform=val_transform, noise_rate=0.0, elastic_num = args.elastic_num)   
        elif args.data_name == 'SFDA_LA':    
            val_ds = SFDA_LA(data_path=data_dir, noise_label_path=args.noise_label_path, data_txt=args.noise_label_txt, mode='test', transform=val_transform)
        elif args.data_name == 'source_LA': 
            val_ds = LA_heart(base_dir='/home/shiteam_public2/seg/data/LASeg/', split='train', num=None, transform=val_transform)
            
        elif args.data_name == 'source_Pancreas': 
            val_ds = Pancreas(base_dir="/home/shiteam_public2/seg/data/Pancreas/", split= 'train', transform=val_transform, noise_rate=0.0, elastic_num = args.elastic_num) 
            
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader





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


'''
# 定义弹性变形参数
transformation = tio.RandomElasticDeformation(
    num_control_points=50,  # 控制点数量（影响变形复杂度）# 之前都是10
    max_displacement=50.0,  # 最大位移量（单位：毫米）
    locked_borders=2,       # 边缘不移动
    image_interpolation='nearest'  # 保持二值标签
)
'''


class MSD_Pancreas(Dataset):
    def __init__(self, json_path, root_dir, mode='train', transform=None):
        """
        胰腺分割数据集加载器
        
        参数:
            json_path (str): JSON配置文件路径
            root_dir (str): 数据根目录
            mode (str): 'train' 或 'test'
            transform (callable): 可选的数据转换/增强
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform if transform else Compose([])
        
        # 加载JSON配置文件
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # 根据模式加载数据路径
        if mode == 'train':
            self.data_list = config['training']
        elif mode == 'test':
            self.data_list = [{'image': path} for path in config['test']]
        else:
            raise ValueError("模式必须是 'train' 或 'test'")
        
        # 存储元数据
        self.name = config['name']
        self.description = config['description']
        self.modality = config['modality']['0']
        self.labels = config['labels']
        self.num_classes = len(self.labels)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        
        # 加载图像
        img_path = os.path.join(self.root_dir, data_item['image'])
        img = nib.load(img_path).get_fdata()
        img = np.expand_dims(img, axis=0)  # 添加通道维度
        '''
        print(f'max:{img.max()}')
        print(f'min:{img.min()}')
        '''
        
        # 对非背景数据进行归一化
        non_background_mask = img != 0
        if np.any(non_background_mask):
            non_background_data = img[non_background_mask]
            min_val = non_background_data.min()
            max_val = non_background_data.max()
            
            if max_val > min_val:  # 避免除以零
                img[non_background_mask] = (non_background_data - min_val) / (max_val - min_val)
        
        # 训练模式加载标签，测试模式只加载图像
        if self.mode == 'train':
            label_path = os.path.join(self.root_dir, data_item['label'])
            label = nib.load(label_path).get_fdata()
            #label = label.astype(np.int64)
            # 将标签值转换为二值：1保持不变，其他值变为0
            label = np.where(label == 1, 1, 0).astype(np.int64)
        else:
            label = np.zeros_like(img[0])  # 测试模式创建空标签
        
        if label.ndim == 3:
            label = np.expand_dims(label, axis=0)
            
        # 应用转换
        sample = {'image': img, 'label': label}
        sample = self.transform(sample)
        
        return sample

class MSD_LA(Dataset):
    def __init__(self, json_path, root_dir, mode='train', transform=None):
        """
        左心房分割数据集加载器
        
        参数:
            json_path (str): JSON配置文件路径
            root_dir (str): 数据根目录
            mode (str): 'train' 或 'test'
            transform (callable): 可选的数据转换/增强
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform if transform else Compose([])
        
        # 加载JSON配置文件
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # 根据模式加载数据路径
        if mode == 'train':
            self.data_list = config['training']
        elif mode == 'test':
            self.data_list = [{'image': path} for path in config['test']]
        else:
            raise ValueError("模式必须是 'train' 或 'test'")
        
        # 存储元数据
        self.name = config['name']
        self.description = config['description']
        self.modality = config['modality']['0']
        self.labels = config['labels']
        self.num_classes = len(self.labels)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        
        # 加载图像
        img_path = os.path.join(self.root_dir, data_item['image'])
        img = nib.load(img_path).get_fdata()
        img = np.expand_dims(img, axis=0)  # 添加通道维度
        
        # 对非背景数据进行归一化
        non_background_mask = img != 0
        if np.any(non_background_mask):
            non_background_data = img[non_background_mask]
            min_val = non_background_data.min()
            max_val = non_background_data.max()
            
            if max_val > min_val:  # 避免除以零
                img[non_background_mask] = (non_background_data - min_val) / (max_val - min_val)
        
        # 训练模式加载标签，测试模式只加载图像
        if self.mode == 'train':
            label_path = os.path.join(self.root_dir, data_item['label'])
            label = nib.load(label_path).get_fdata()
            label = label.astype(np.int64)
        else:
            label = np.zeros_like(img[0])  # 测试模式创建空标签
        
        if label.ndim == 3:
            label = np.expand_dims(label, axis=0)
            
        # 应用转换
        sample = {'image': img, 'label': label}
        sample = self.transform(sample)
        
        return sample
        
class SFDA_LA(Dataset):
    def __init__(self, data_path, noise_label_path, data_txt, mode='train', transform=None):
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        
        # 读取病例列表
        try:
            with open(data_txt, 'r') as f:
                case_names = set(line.strip() for line in f.readlines() if line.strip())
        except FileNotFoundError:
            raise FileNotFoundError(f"无法找到数据列表文件: {data_txt}")
        
        # 重构路径加载逻辑
        if mode == 'train':
            train_dir = os.path.join(data_path, 'Training')
            for case in os.listdir(train_dir):
                case_dir = os.path.join(train_dir, case)
                if not os.path.isdir(case_dir):
                    continue
                    
                # 核心改进：成对添加路径
                self.image_paths.append(case_dir)
                
                # 决定标签路径类型
                if case in case_names:
                    noise_path = os.path.join(noise_label_path, case)
                    if os.path.isdir(noise_path):
                        self.label_paths.append(noise_path)
                    else:
                        print(f"警告: 噪声标签路径不存在，使用原始标签: {case}")
                        self.label_paths.append(case_dir)
                else:
                    self.label_paths.append(case_dir)
        
        
        elif mode == 'test':
            file_path = os.path.join(data_path, 'Testing')
            self.image_paths = []
            self.label_paths = []
            for case in os.listdir(file_path):
                if os.path.isdir(os.path.join(file_path, case)):
                    self.image_paths.append(os.path.join(file_path, case)) 
                    self.label_paths.append(os.path.join(file_path, case))
        '''
        elif mode == 'test':
            test_dir = os.path.join(data_path, 'Testing')
            for case in os.listdir(test_dir):
                case_dir = os.path.join(test_dir, case)
                if os.path.isdir(case_dir):
                    self.image_paths.append(case_dir)
                    self.label_paths.append(case_dir)  # 测试模式使用相同路径
        '''
        # 关键修复：验证路径匹配性
        if len(self.image_paths) != len(self.label_paths):
            raise RuntimeError(
                f"路径数量不匹配! 图像路径: {len(self.image_paths)}, 标签路径: {len(self.label_paths)}"
            )
        
        # 添加调试信息
        print(f"成功初始化 {mode} 数据集: {len(self.image_paths)} 个样本")
        if mode == 'train':
            print(f"使用噪声标签的病例: {len(case_names & set(os.path.basename(p) for p in self.label_paths))}")

    def __getitem__(self, item):
        # 确保使用相同索引获取对应路径
        image_dir = self.image_paths[item]
        label_dir = self.label_paths[item]
        
        # 统一路径构建方式
        image_path = os.path.join(image_dir, 'mri.nii.gz')
        label_path = os.path.join(label_dir, 'endo.nii.gz')
        
        if not os.path.isfile(label_path):
            alt_label = os.path.join(label_dir, 'pred.nii.gz')
            if os.path.isfile(alt_label):
                label_path = alt_label
            else:
                raise FileNotFoundError(f"在 {label_dir} 中找不到标签文件")
        '''
        print(image_path)
        print(label_path)
        '''
        
        # 添加形状验证
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        
        if image.shape != label.shape:
            raise RuntimeError(
                f"形状不匹配! {image_path}: {image.shape} vs {label_path}: {label.shape}"
            )
    
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
        return len(self.image_paths)
    
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
               
        self.mode = mode
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
        if random_number < self.noise_rate and self.mode == 'Training':
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


class LA_heart(Dataset):
    """ 2018 LA_seg Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        
        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/2018LA_Seg_Training Set/" + image_name + "/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        if label.ndim == 3:
            label = np.expand_dims(label, axis=0) 
        
        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample




'''
class Pancreas(Dataset):
    """ Pancreas Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir + '/train.list'
        test_path = self._base_dir + '/test.list'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
       image_name = self.image_list[idx]
       h5f = h5py.File(self._base_dir + "/pancreas_data/" + image_name + ".h5", 'r')
       image = h5f['image'][:]
       label = h5f['label'][:]
       if image.ndim == 3:
           image = np.expand_dims(image, axis=0) 
       if label.ndim == 3:
           label = np.expand_dims(label, axis=0)  
       sample = {'image': image, 'label': label}
       #sample = ('image': image, 'label': label)
       if self.transform:
           sample = self.transform(sample)
           image = sample['image']
           label = sample['label']
       return (image,label) 
'''

class Pancreas(Dataset):
    """ Pancreas Dataset with elastic deformation noise """
    
    def __init__(self, base_dir=None, split='train', num=None, transform=None, noise_rate=0.0, elastic_num=10):
        """
        Args:
            base_dir (str): 数据集根目录
            split (str): 'train' 或 'test'
            num (int, optional): 使用的样本数量
            transform (callable, optional): 数据转换函数
            noise_rate (float): 添加弹性形变噪声的概率 (0.0 - 1.0)
            elastic_num (int): 控制弹性形变的点数量 (影响形变程度)
        """
        self._base_dir = base_dir
        self.transform = transform
        self.noise_rate = noise_rate
        self.elastic_num = elastic_num
        self.split = split
        
        # 定义弹性形变转换器
        self.elastic_transform = tio.RandomElasticDeformation(
            num_control_points=elastic_num,  # 控制点数量
            max_displacement=float(elastic_num),  # 最大位移量（单位：毫米）
            locked_borders=2,  # 边缘不移动
            image_interpolation='nearest'  # 保持二值标签 (最近邻插值)
        )
        
        self.sample_list = []
        train_path = self._base_dir + '/train.list'
        test_path = self._base_dir + '/test.list'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print(f"total {len(self.image_list)} samples, noise rate: {noise_rate}, elastic points: {elastic_num}")

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/pancreas_data/" + image_name + ".h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        
        # 确保数据是三维格式 (C, D, H, W)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0) 
        if label.ndim == 3:
            label = np.expand_dims(label, axis=0)
        
        
        # 创建样本字典
        sample = {'image': image, 'label': label}
        
        # 决定是否应用弹性形变噪声 (只针对标签)
        if self.noise_rate > 0.0 and random.random() < self.noise_rate and self.split == 'train':
            # 只对标签应用弹性形变，不影响原始图像
            # 转换为torch张量，并添加通道维度以符合torchio要求
            label_tensor = torch.from_numpy(label)
            
            # 创建包含标签的Subject对象
            subject = tio.Subject(
                label=tio.LabelMap(tensor=label_tensor)
            )
            
            # 应用弹性形变
            transformed_subject = self.elastic_transform(subject)
            
            # 提取变形后的标签数据并恢复为NumPy数组
            deformed_label = transformed_subject.label.data.numpy()
            
            # 确保变形后的标签形状与原始标签一致
            if deformed_label.shape == label.shape:
                sample['label'] = deformed_label
                #print('aa')
            else:
                print(f"Warning: Shape mismatch after deformation {deformed_label.shape} vs {label.shape}. Using original label.")
        
        
        # 应用额外的转换 (如果有)
        if self.transform:
            sample = self.transform(sample)
        
        return sample['image'], sample['label']




class BraTS_noise(Dataset):
    def __init__(self, file_list, transform=None, noise_rate=0.0, elastic_num=10):
        self.file_list = file_list
        self.transform = transform
        self.noise_rate = noise_rate
        # 定义弹性形变转换器
        self.elastic_transform = tio.RandomElasticDeformation(
            num_control_points=elastic_num,  # 控制点数量
            max_displacement=float(elastic_num),  # 最大位移量（单位：毫米）
            locked_borders=2,  # 边缘不移动
            image_interpolation='nearest'  # 保持二值标签 (最近邻插值)
        )

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
           transformed_subject = self.elastic_transform(subject)
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
        # ET, WT, TC
        '''
        label_onehot[0] = (label == 4).astype(np.uint8)
        label_onehot[1] = np.logical_or(label == 1, label == 4, label == 2).astype(np.uint8)
        label_onehot[2] = np.logical_or(label == 1, label == 4).astype(np.uint8)
        '''
        
        
        label_onehot[0] = np.logical_or(label == 1, label == 4).astype(np.uint8)
        label_onehot[1] = np.logical_or(label == 1, label == 4, label == 2).astype(np.uint8)
        label_onehot[2] = (label == 4).astype(np.uint8)
        

        # 如果有变换，应用变换
        if self.transform:
            sample = self.transform({'image': image, 'label': label_onehot})
            image = sample['image']
            label = sample['label']
           
        return (image,label)
    
    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
        
        

'''
class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        k = np.random.randint(0, 4)
        image = np.stack([np.rot90(x,k) for x in image],axis=0)
        label = np.rot90(label, k)
        axis = np.random.randint(1, 4)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}
'''

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


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


class GaussianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)
        return {'image': image, 'label': label}
def augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
    if not per_channel:
        mn = data_sample.mean()
        if preserve_range:
            minm = data_sample.min()
            maxm = data_sample.max()
        if np.random.random() < 0.5 and contrast_range[0] < 1:
            factor = np.random.uniform(contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
        data_sample = (data_sample - mn) * factor + mn
        if preserve_range:
            data_sample[data_sample < minm] = minm
            data_sample[data_sample > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            mn = data_sample[c].mean()
            if preserve_range:
                minm = data_sample[c].min()
                maxm = data_sample[c].max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data_sample[c] = (data_sample[c] - mn) * factor + mn
            if preserve_range:
                data_sample[c][data_sample[c] < minm] = minm
                data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


class ContrastAugmentationTransform(object):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True,p_per_sample=1.):
        self.p_per_sample = p_per_sample
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        for b in range(len(image)):
            if np.random.uniform() < self.p_per_sample:
                image[b] = augment_contrast(image[b], contrast_range=self.contrast_range,
                                            preserve_range=self.preserve_range, per_channel=self.per_channel)
        return {'image': image, 'label': label}

def augment_brightness_additive(data_sample, mu:float, sigma:float , per_channel:bool=True, p_per_channel:float=1.):
    if not per_channel:
        rnd_nb = np.random.normal(mu, sigma)
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                data_sample[c] += rnd_nb
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                rnd_nb = np.random.normal(mu, sigma)
                data_sample[c] += rnd_nb
    return data_sample


class BrightnessTransform(object):
    def __init__(self, mu, sigma, per_channel=True, p_per_sample=1., p_per_channel=1.):
        self.p_per_sample = p_per_sample
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, sample):
        data, label = sample['image'], sample['label']

        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data[b] = augment_brightness_additive(data[b], self.mu, self.sigma, self.per_channel,
                                                      p_per_channel=self.p_per_channel)

        return {'image': data, 'label': label}

'''

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
'''



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





