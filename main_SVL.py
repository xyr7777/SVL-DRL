import warnings  #忽略警告
import argparse
import os
from functools import partial
from utils.reproducibility import set_global_seed
import itertools

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
#from trainer import run_training
#from trainer_noise import run_training_noise
#from trainer_DL import run_training

#from utils.dataset_noise import get_loader
from utils.dataset_new import get_loader
#from utils.data_utils import get_loader
#from data.pre_dataset import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

from monai.networks.nets import UNet
from trainer_v9 import run_training
from policy_value_v7 import PolicyValueSegmentationNetwork
from swin_unetr_policy_value import SwinUNETR_policy_value
from typing import Callable, Dict, List, Tuple

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline for BRATS Challenge")
parser.add_argument("--data_name", default='MT_Pancreas', type=str, help="training for which dataset")
# Pancreas, left_atrial, Brats21, SFDA_LA, SFDA_Pancreas, SFDA_Brats21,  MT_Pancreas, MT_LA
parser.add_argument("--seed", default=42, type=int, help="base random seed")
parser.add_argument("--deterministic", action="store_true", help="enable deterministic PyTorch operations (may reduce performance)")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="MT_Pancreas_noise0.5", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--fold", default=1, type=int, help="data fold")
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--voxelspacing", default=1.5, type=float, help="image spacing")

parser.add_argument("--data_dir", default="/home/shiteam_public2/seg/data/LA_heart_3D_nii/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="/home/amsshi/fyy/data/left_atrial/dataset.json", type=str, help="dataset json file")
parser.add_argument("--noise_label_txt", default="/home/shiteam_public2/seg/code/SwinUNETR/LA/data/target_0.5.txt", type=str, help="train_txt directory")
parser.add_argument("--noise_label_path", default="/home/shiteam_public2/seg/code/SwinUNETR/LA/predictions/unet_source20_target80/", type=str, help="dataset json file")

parser.add_argument("--noise_rate", default=0.5, type=float, help="noise_rate")
parser.add_argument("--elastic_num", default=10, type=float, help="elastic_num")

parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--start_epoch", default=199, type=int, help="train from epoch")
parser.add_argument("--max_epochs", default=15000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")

parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")

parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--cache_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=112, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=112, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=80, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="cosine_anneal", type=str, help="type of learning rate scheduler")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument(
    "--pretrained_dir",
    default="/home/shiteam_public2/seg/code/SVL-DRL/LA/runs_SVL/MT_Pancreas_noise0.5/",
    type=str,
    help="pretrained checkpoint directory",
)
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--MODEL.RESUME", action="store_true", help="resume from checkpoint")

parser.add_argument("--warmup_epochs", default=200, type=int, help="number of warmup epochs")

# 在训练参数中添加
parser.add_argument('--n_steps', type=int, default=3, help="TD步数")
parser.add_argument('--gamma', type=float, default=0.99, help="折扣因子")
parser.add_argument('--gae_lambda', type=float, default=0.95, help="GAE参数")
parser.add_argument('--t_start', type=int, default=0, help="开始步次")
parser.add_argument('--t_max', type=int, default=3, help="结束步次")
parser.add_argument('--action_classes', type=int, default=3, help="动作类别数")

# 建议在参数解析器中添加
parser.add_argument('--policy_lr', type=float, default=1e-4)
parser.add_argument('--value_lr', type=float, default=1e-4)
parser.add_argument('--seg_lr', type=float, default=1e-4)
parser.add_argument('--policy_reg', type=float, default=1e-5)
parser.add_argument('--value_reg', type=float, default=1e-5)
parser.add_argument('--seg_reg', type=float, default=1e-5)

def main():
    # 如果你想忽略所有警告，可以这样设置
    warnings.filterwarnings("ignore")
    
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs_SVL/" + args.logdir
    args.save_checkpoint = True
    # args.MODEL.RESUME = True
    # 是否导入预训练模型
    args.resume_ckpt = True
    
    if args.data_name == 'left_atrial':
        args.data_dir = "/home/shiteam_public2/seg/data/LA_heart_3D_nii/"
        args.roi_x = 96
        args.roi_y = 96
        args.roi_z = 64
        args.in_channels = 1
        args.out_channels = 1
        args.voxelspacing = 1.0
    elif args.data_name == 'Pancreas':
        args.data_dir = "/home/shiteam_public2/seg/data/Pancreas/"
        args.roi_x = 96
        args.roi_y = 96
        args.roi_z = 96
        args.in_channels = 1
        args.out_channels = 1
        args.voxelspacing = 1.0
    elif args.data_name == 'Brats21':
        args.data_dir = "/home/shiteam_public2/seg/data/brats-2021-task1/"
        args.roi_x = 128
        args.roi_y = 128
        args.roi_z = 128
        args.in_channels = 4
        args.out_channels = 3
        args.voxelspacing = 1.0
        args.optim_name = 'adamw'
        args.optim_lr = 1e-4
    elif args.data_name == 'SFDA_LA':
        args.data_dir = "/home/shiteam_public2/seg/data/LA_heart_3D_nii/"
        args.roi_x = 96
        args.roi_y = 96
        args.roi_z = 64
        args.in_channels = 1
        args.out_channels = 1
        args.voxelspacing = 1.0
    
    elif args.data_name == 'MT_Pancreas':
        args.data_dir = "/home/shiteam_public2/seg/data/Pancreas/"
        args.roi_x = 96
        args.roi_y = 96
        args.roi_z = 96
        args.in_channels = 1
        args.out_channels = 1
        args.voxelspacing = 1.0
    
    elif args.data_name == 'MT_LA':
        args.data_dir = "/home/shiteam_public2/seg/data/LASeg/"
        args.roi_x = 96
        args.roi_y = 96
        args.roi_z = 96
        args.in_channels = 1
        args.out_channels = 1
        args.voxelspacing = 0.625
    
    
    
    
    
    
    
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = not args.deterministic
    args.test_mode = False
    
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    
    
    model = SwinUNETR_policy_value(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
    )
    
    


    if args.resume_ckpt:
        model_dict = torch.load(pretrained_pth)["state_dict"]
        model.load_state_dict(model_dict)
        print("Using pretrained weights")

    if args.squared_dice:
        dice_loss = DiceLoss(
            to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    # post_pred = AsDiscrete(argmax=False, logit_thresh=0.5)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    
    
    '''
    model = PolicyValueSegmentationNetwork(
            in_channels=args.in_channels,
            seg_channels=args.out_channels,
            #spatial_dims=args.spatial_dims
        )
    
    '''
    
    '''
    model = SwinUNETR_policy_value(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
    )
    '''
    
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model.forward_segmentation_only,
        overlap=args.infer_overlap,
    )
    
    
    
    
    best_acc = 0
    start_epoch = args.start_epoch

    if args.checkpoint is not None:
        
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
        

    # 根据SwinUNETR模型结构和参数分组θ_p, θ_v, θ_s修改优化器配置
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam([
            # 策略网络参数 (θ_p) - policy路径
            {
                'params': itertools.chain(
                    model.policy_decoder.parameters(),
                    model.policy_up1.parameters(),
                    model.policy_up2.parameters(),
                    model.policy_up3.parameters(),
                    model.policy_up4.parameters(),
                    model.policy_out.parameters()
                ),
                'lr': args.policy_lr,
                'weight_decay': args.policy_reg
            },
            # 价值网络参数 (θ_v) - value路径
            {
                'params': itertools.chain(
                    model.value_encoder1.parameters(),
                    model.value_encoder2.parameters(),
                    model.value_decoder.parameters()
                ),
                'lr': args.value_lr,
                'weight_decay': args.value_reg
            },
            # 分割网络参数 (θ_s) - SwinViT主干 + 分割解码器
            {
                'params': itertools.chain(
                    model.swinViT.parameters(),
                    model.encoder1.parameters(),
                    model.encoder2.parameters(),
                    model.encoder3.parameters(),
                    model.encoder4.parameters(),
                    model.encoder10.parameters(),
                    model.decoder5.parameters(),
                    model.decoder4.parameters(),
                    model.decoder3.parameters(),
                    model.decoder2.parameters(),
                    model.decoder1.parameters(),
                    model.seg_out.parameters()
                ),
                'lr': args.seg_lr,
                'weight_decay': args.seg_reg
            }
        ])
        
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW([
            # 策略网络
            {
                'params': itertools.chain(
                    model.policy_decoder.parameters(),
                    model.policy_up1.parameters(),
                    model.policy_up2.parameters(),
                    model.policy_up3.parameters(),
                    model.policy_up4.parameters(),
                    model.policy_out.parameters()
                ),
                'lr': args.policy_lr,
                'weight_decay': args.policy_reg
            },
            # 价值网络
            {
                'params': itertools.chain(
                    model.value_encoder1.parameters(),
                    model.value_encoder2.parameters(),
                    model.value_decoder.parameters()
                ),
                'lr': args.value_lr,
                'weight_decay': args.value_reg
            },
            # 分割网络
            {
                'params': itertools.chain(
                    model.swinViT.parameters(),
                    model.encoder1.parameters(),
                    model.encoder2.parameters(),
                    model.encoder3.parameters(),
                    model.encoder4.parameters(),
                    model.encoder10.parameters(),
                    model.decoder5.parameters(),
                    model.decoder4.parameters(),
                    model.decoder3.parameters(),
                    model.decoder2.parameters(),
                    model.decoder1.parameters(),
                    model.seg_out.parameters()
                ),
                'lr': args.seg_lr,
                'weight_decay': args.seg_reg
            }
        ])
        
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD([
            # 策略网络
            {
                'params': itertools.chain(
                    model.policy_decoder.parameters(),
                    model.policy_up1.parameters(),
                    model.policy_up2.parameters(),
                    model.policy_up3.parameters(),
                    model.policy_up4.parameters(),
                    model.policy_out.parameters()
                ),
                'lr': args.policy_lr,
                'momentum': args.momentum,
                'nesterov': True,
                'weight_decay': args.policy_reg
            },
            # 价值网络
            {
                'params': itertools.chain(
                    model.value_encoder1.parameters(),
                    model.value_encoder2.parameters(),
                    model.value_decoder.parameters()
                ),
                'lr': args.value_lr,
                'momentum': args.momentum,
                'nesterov': True,
                'weight_decay': args.value_reg
            },
            # 分割网络
            {
                'params': itertools.chain(
                    model.swinViT.parameters(),
                    model.encoder1.parameters(),
                    model.encoder2.parameters(),
                    model.encoder3.parameters(),
                    model.encoder4.parameters(),
                    model.encoder10.parameters(),
                    model.decoder5.parameters(),
                    model.decoder4.parameters(),
                    model.decoder3.parameters(),
                    model.decoder2.parameters(),
                    model.decoder1.parameters(),
                    model.seg_out.parameters()
                ),
                'lr': args.seg_lr,
                'momentum': args.momentum,
                'nesterov': True,
                'weight_decay': args.seg_reg
            }
        ])
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))
    
    
    # 学习率调度器（保持不变）
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.max_epochs
        )    
        
    semantic_classes = ['heart']
    accuracy = run_training(
        model = model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
        semantic_classes=semantic_classes,
    )
    return accuracy


if __name__ == "__main__":
    main()
