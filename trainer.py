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

import os
import pdb
import shutil
import time
import warnings  #忽略警告
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import *
# from utils.utils import AverageMeter, distributed_all_gather
from monai.data import decollate_batch





def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        
        '''
        if idx >= 1:
           break
        '''
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
            
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        
        '''
        print(data.size())
        print(target.size())
        '''
        
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg

'''
def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()
    

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            
            with autocast(enabled=args.amp):
                logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                Dice = run_acc.avg[0]
                
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    ", Dice:",
                    Dice,
                    ", time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()

    return run_acc.avg





def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    semantic_classes=None,
):
    writer = True
    # 如果你想忽略所有警告，可以这样设置
    warnings.filterwarnings("ignore")
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    train_file = os.path.join(args.logdir, 'train_log.txt')
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            info1 = ("epoch:{} ,train_loss:{}".format(epoch, train_loss))
            
            with open(train_file,'a') as f:
                f.write(info1 + '\n')
                
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            if args.rank == 0:
                Dice = val_acc[0]
                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    ", Dice:",
                    Dice,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )

                if writer is not None:
                    writer.add_scalar("Mean_Val_Dice", np.mean(val_acc), epoch)
                    
                    info2 = (
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1)+
                    ", Dice:{:.2f}".format(
                    Dice) +
                    ", time {:.2f}s".format(time.time() - epoch_time)
                )
                    
                    with open(train_file,'a') as f:
                        f.write(info2 + '\n' )
                    if semantic_classes is not None:
                        for val_channel_ind in range(len(semantic_classes)):
                            if val_channel_ind < val_acc.size:
                                writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
                val_avg_acc = np.mean(val_acc)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            info4 = ("best_Val_Dice:{}".format(val_acc_max))            
            with open(train_file,'a') as f:
                f.write(info4 + '\n')
            
                
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))   

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
'''

'''
def val_epoch(model, loader, epoch, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    
    # 使用自定义指标计算器
    metrics_calculator = CustomMetricsCalculator(voxelspacing=args.voxelspacing)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            
            with autocast(enabled=args.amp):
                logits = model_inferer(data)
            
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            
            # 重置指标计算器
            metrics_calculator.reset()
            
            # 计算每个样本的指标
            for pred, label in zip(val_output_convert, val_labels_list):
                # 确保数据格式正确
                pred = pred.float()
                label = label.float()
                
                # 计算指标
                metrics_calculator(pred, label)
            
            # 聚合结果
            dice_avg, jaccard_avg, hd95_avg, asd_avg = metrics_calculator.aggregate()
            
            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    ", Dice: {:.4f}".format(dice_avg),
                    ", Jaccard: {:.4f}".format(jaccard_avg),
                    ", HD95: {:.4f}".format(hd95_avg),
                    ", ASD: {:.4f}".format(asd_avg),
                    ", time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()

    # 返回所有指标的平均值
    return dice_avg, jaccard_avg, hd95_avg, asd_avg
'''




def val_epoch(model, loader, epoch, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    
    # 使用自定义指标计算器
    metrics_calculator = CustomMetricsCalculator(voxelspacing=args.voxelspacing)
    
    # 初始化指标累积器
    all_dice = []
    all_jaccard = []
    all_hd95 = []
    all_asd = []
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            
            with autocast(enabled=args.amp):
                logits = model_inferer(data)
            
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            
            # 重置指标计算器
            metrics_calculator.reset()
            
            # 计算每个样本的指标
            for pred, label in zip(val_output_convert, val_labels_list):
                # 确保数据格式正确
                if isinstance(pred, torch.Tensor):
                    pred = pred.detach().cpu().numpy()
                if isinstance(label, torch.Tensor):
                    label = label.detach().cpu().numpy()
                
                # 计算指标
                dice, jaccard, hd95, asd = metrics_calculator(pred, label)
                
                # 累积指标
                all_dice.append(dice)
                all_jaccard.append(jaccard)
                all_hd95.append(hd95)
                all_asd.append(asd)
            
            # 计算当前batch的平均指标
            batch_dice = np.mean(all_dice[-len(val_labels_list):])
            batch_jaccard = np.mean(all_jaccard[-len(val_labels_list):])
            batch_hd95 = np.mean(all_hd95[-len(val_labels_list):])
            batch_asd = np.mean(all_asd[-len(val_labels_list):])
            
            if args.rank == 0:
                print(
                    f"Val {epoch}/{args.max_epochs} batch {idx}/{len(loader)}",
                    f"Dice: {batch_dice:.4f}",
                    f"Jaccard: {batch_jaccard:.4f}",
                    f"HD95: {batch_hd95:.4f}",
                    f"ASD: {batch_asd:.4f}",
                    f"time: {time.time() - start_time:.2f}s"
                )
            start_time = time.time()
    
    # 计算整个验证集的平均指标
    mean_dice = np.mean(all_dice) if all_dice else 0.0
    mean_jaccard = np.mean(all_jaccard) if all_jaccard else 0.0
    mean_hd95 = np.mean(all_hd95) if all_hd95 else 0.0
    mean_asd = np.mean(all_asd) if all_asd else 0.0
    
    if args.rank == 0:
        print(
            f"Validation completed for epoch {epoch}",
            f"Mean Dice: {mean_dice:.4f}",
            f"Mean Jaccard: {mean_jaccard:.4f}",
            f"Mean HD95: {mean_hd95:.4f}",
            f"Mean ASD: {mean_asd:.4f}"
        )
    
    return mean_dice, mean_jaccard, mean_hd95, mean_asd



def val_epoch_brats21(model, loader, epoch, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    
    # 使用自定义指标计算器
    metrics_calculator = CustomMetricsCalculator(voxelspacing=args.voxelspacing)
    
    # 初始化指标累积器 - 为每个标签创建单独的列表
    all_dice = {'ET': [], 'WT': [], 'TC': [], 'mean': []}
    all_jaccard = {'ET': [], 'WT': [], 'TC': [], 'mean': []}
    all_hd95 = {'ET': [], 'WT': [], 'TC': [], 'mean': []}
    all_asd = {'ET': [], 'WT': [], 'TC': [], 'mean': []}
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            
            with autocast(enabled=args.amp):
                logits = model_inferer(data)
            
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            
            # 计算每个样本的指标
            for pred, label in zip(val_output_convert, val_labels_list):
                # 确保数据格式正确
                if isinstance(pred, torch.Tensor):
                    pred = pred.detach().cpu().numpy()
                if isinstance(label, torch.Tensor):
                    label = label.detach().cpu().numpy()
                
                # 计算每个标签的指标
                dice_et, jaccard_et, hd95_et, asd_et = metrics_calculator(pred[0], label[0])  # ET
                dice_wt, jaccard_wt, hd95_wt, asd_wt = metrics_calculator(pred[1], label[1])  # WT
                dice_tc, jaccard_tc, hd95_tc, asd_tc = metrics_calculator(pred[2], label[2])  # TC
                
                # 计算平均指标
                dice_mean = (dice_et + dice_wt + dice_tc) / 3
                jaccard_mean = (jaccard_et + jaccard_wt + jaccard_tc) / 3
                hd95_mean = (hd95_et + hd95_wt + hd95_tc) / 3
                asd_mean = (asd_et + asd_wt + asd_tc) / 3
                
                # 累积指标
                all_dice['ET'].append(dice_et)
                all_dice['WT'].append(dice_wt)
                all_dice['TC'].append(dice_tc)
                all_dice['mean'].append(dice_mean)
                
                all_jaccard['ET'].append(jaccard_et)
                all_jaccard['WT'].append(jaccard_wt)
                all_jaccard['TC'].append(jaccard_tc)
                all_jaccard['mean'].append(jaccard_mean)
                
                all_hd95['ET'].append(hd95_et)
                all_hd95['WT'].append(hd95_wt)
                all_hd95['TC'].append(hd95_tc)
                all_hd95['mean'].append(hd95_mean)
                
                all_asd['ET'].append(asd_et)
                all_asd['WT'].append(asd_wt)
                all_asd['TC'].append(asd_tc)
                all_asd['mean'].append(asd_mean)
            
            # 计算当前batch的平均指标
            batch_dice_et = np.mean(all_dice['ET'][-len(val_labels_list):])
            batch_dice_wt = np.mean(all_dice['WT'][-len(val_labels_list):])
            batch_dice_tc = np.mean(all_dice['TC'][-len(val_labels_list):])
            batch_dice_mean = np.mean(all_dice['mean'][-len(val_labels_list):])
            
            batch_jaccard_et = np.mean(all_jaccard['ET'][-len(val_labels_list):])
            batch_jaccard_wt = np.mean(all_jaccard['WT'][-len(val_labels_list):])
            batch_jaccard_tc = np.mean(all_jaccard['TC'][-len(val_labels_list):])
            batch_jaccard_mean = np.mean(all_jaccard['mean'][-len(val_labels_list):])
            
            batch_hd95_et = np.mean(all_hd95['ET'][-len(val_labels_list):])
            batch_hd95_wt = np.mean(all_hd95['WT'][-len(val_labels_list):])
            batch_hd95_tc = np.mean(all_hd95['TC'][-len(val_labels_list):])
            batch_hd95_mean = np.mean(all_hd95['mean'][-len(val_labels_list):])
            
            batch_asd_et = np.mean(all_asd['ET'][-len(val_labels_list):])
            batch_asd_wt = np.mean(all_asd['WT'][-len(val_labels_list):])
            batch_asd_tc = np.mean(all_asd['TC'][-len(val_labels_list):])
            batch_asd_mean = np.mean(all_asd['mean'][-len(val_labels_list):])
            
            if args.rank == 0:
                print(
                    f"Val {epoch}/{args.max_epochs} batch {idx}/{len(loader)}",
                    f"Dice - ET: {batch_dice_et:.4f}, WT: {batch_dice_wt:.4f}, TC: {batch_dice_tc:.4f}, Mean: {batch_dice_mean:.4f}",
                    f"Jaccard - ET: {batch_jaccard_et:.4f}, WT: {batch_jaccard_wt:.4f}, TC: {batch_jaccard_tc:.4f}, Mean: {batch_jaccard_mean:.4f}",
                    f"HD95 - ET: {batch_hd95_et:.4f}, WT: {batch_hd95_wt:.4f}, TC: {batch_hd95_tc:.4f}, Mean: {batch_hd95_mean:.4f}",
                    f"ASD - ET: {batch_asd_et:.4f}, WT: {batch_asd_wt:.4f}, TC: {batch_asd_tc:.4f}, Mean: {batch_asd_mean:.4f}",
                    f"time: {time.time() - start_time:.2f}s"
                )
            start_time = time.time()
    
    # 计算整个验证集的平均指标
    mean_dice_et = np.mean(all_dice['ET']) if all_dice['ET'] else 0.0
    mean_dice_wt = np.mean(all_dice['WT']) if all_dice['WT'] else 0.0
    mean_dice_tc = np.mean(all_dice['TC']) if all_dice['TC'] else 0.0
    mean_dice_mean = np.mean(all_dice['mean']) if all_dice['mean'] else 0.0
    
    mean_jaccard_et = np.mean(all_jaccard['ET']) if all_jaccard['ET'] else 0.0
    mean_jaccard_wt = np.mean(all_jaccard['WT']) if all_jaccard['WT'] else 0.0
    mean_jaccard_tc = np.mean(all_jaccard['TC']) if all_jaccard['TC'] else 0.0
    mean_jaccard_mean = np.mean(all_jaccard['mean']) if all_jaccard['mean'] else 0.0
    
    mean_hd95_et = np.mean(all_hd95['ET']) if all_hd95['ET'] else 0.0
    mean_hd95_wt = np.mean(all_hd95['WT']) if all_hd95['WT'] else 0.0
    mean_hd95_tc = np.mean(all_hd95['TC']) if all_hd95['TC'] else 0.0
    mean_hd95_mean = np.mean(all_hd95['mean']) if all_hd95['mean'] else 0.0
    
    mean_asd_et = np.mean(all_asd['ET']) if all_asd['ET'] else 0.0
    mean_asd_wt = np.mean(all_asd['WT']) if all_asd['WT'] else 0.0
    mean_asd_tc = np.mean(all_asd['TC']) if all_asd['TC'] else 0.0
    mean_asd_mean = np.mean(all_asd['mean']) if all_asd['mean'] else 0.0
    
    if args.rank == 0:
        print("\n" + "="*80)
        print(f"Validation completed for epoch {epoch}")
        print("="*80)
        print(f"Dice Scores:")
        print(f"  ET: {mean_dice_et:.4f}")
        print(f"  WT: {mean_dice_wt:.4f}")
        print(f"  TC: {mean_dice_tc:.4f}")
        print(f"  Mean: {mean_dice_mean:.4f}")
        
        print(f"\nJaccard Scores:")
        print(f"  ET: {mean_jaccard_et:.4f}")
        print(f"  WT: {mean_jaccard_wt:.4f}")
        print(f"  TC: {mean_jaccard_tc:.4f}")
        print(f"  Mean: {mean_jaccard_mean:.4f}")
        
        print(f"\nHausdorff Distance (95%):")
        print(f"  ET: {mean_hd95_et:.4f}")
        print(f"  WT: {mean_hd95_wt:.4f}")
        print(f"  TC: {mean_hd95_tc:.4f}")
        print(f"  Mean: {mean_hd95_mean:.4f}")
        
        print(f"\nAverage Surface Distance:")
        print(f"  ET: {mean_asd_et:.4f}")
        print(f"  WT: {mean_asd_wt:.4f}")
        print(f"  TC: {mean_asd_tc:.4f}")
        print(f"  Mean: {mean_asd_mean:.4f}")
        print("="*80)
    
    # 返回所有指标
    return {
        'dice': {
            'ET': mean_dice_et,
            'WT': mean_dice_wt,
            'TC': mean_dice_tc,
            'mean': mean_dice_mean
        },
        'jaccard': {
            'ET': mean_jaccard_et,
            'WT': mean_jaccard_wt,
            'TC': mean_jaccard_tc,
            'mean': mean_jaccard_mean
        },
        'hd95': {
            'ET': mean_hd95_et,
            'WT': mean_hd95_wt,
            'TC': mean_hd95_tc,
            'mean': mean_hd95_mean
        },
        'asd': {
            'ET': mean_asd_et,
            'WT': mean_asd_wt,
            'TC': mean_asd_tc,
            'mean': mean_asd_mean
        }
    }



def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    semantic_classes=None,
):
    writer = True
    warnings.filterwarnings("ignore")
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_dice_max = 0.0
    train_file = os.path.join(args.logdir, 'train_log.txt')
    
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.time(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            info1 = "epoch:{} ,train_loss:{}".format(epoch, train_loss)
            
            with open(train_file, 'a') as f:
                f.write(info1 + '\n')
                
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            
            # 调用修改后的val_epoch函数
            dice_avg, jaccard_avg, hd95_avg, asd_avg = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                args=args,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            if args.rank == 0:
                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    ", Dice: {:.4f}".format(dice_avg),
                    ", Jaccard: {:.4f}".format(jaccard_avg),
                    ", HD95: {:.4f}".format(hd95_avg),
                    ", ASD: {:.4f}".format(asd_avg),
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )

                if writer is not None:
                    writer.add_scalar("Val_Dice", dice_avg, epoch)
                    writer.add_scalar("Val_Jaccard", jaccard_avg, epoch)
                    writer.add_scalar("Val_HD95", hd95_avg, epoch)
                    writer.add_scalar("Val_ASD", asd_avg, epoch)
                    
                    info2 = (
                        "Final validation stats {}/{}".format(epoch, args.max_epochs - 1) +
                        ", Dice: {:.4f}".format(dice_avg) +
                        ", Jaccard: {:.4f}".format(jaccard_avg) +
                        ", HD95: {:.4f}".format(hd95_avg) +
                        ", ASD: {:.4f}".format(asd_avg) +
                        ", time {:.2f}s".format(time.time() - epoch_time)
                    )
                    
                    with open(train_file, 'a') as f:
                        f.write(info2 + '\n')
                
                # 使用Dice作为选择最佳模型的标准
                if dice_avg > val_dice_max:
                    print("new best Dice ({:.6f} --> {:.6f}). ".format(val_dice_max, dice_avg))
                    val_dice_max = dice_avg
                    b_new_best = True
                    
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_dice_max, optimizer=optimizer, scheduler=scheduler
                        )
            
            info4 = "best_Val_Dice:{}".format(val_dice_max)            
            with open(train_file, 'a') as f:
                f.write(info4 + '\n')
            
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_dice_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))   

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Dice Accuracy: ", val_dice_max)
    return val_dice_max


def run_training_brats21(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    semantic_classes=None,
):
    writer = True
    warnings.filterwarnings("ignore")
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    
    # 使用字典存储最佳指标
    best_metrics = {
        'dice': {'ET': 0.0, 'WT': 0.0, 'TC': 0.0, 'mean': 0.0},
        'jaccard': {'ET': 0.0, 'WT': 0.0, 'TC': 0.0, 'mean': 0.0},
        'hd95': {'ET': 0.0, 'WT': 0.0, 'TC': 0.0, 'mean': 0.0},
        'asd': {'ET': 0.0, 'WT': 0.0, 'TC': 0.0, 'mean': 0.0}
    }
    
    train_file = os.path.join(args.logdir, 'train_log.txt')
    
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            info1 = "epoch:{} ,train_loss:{}".format(epoch, train_loss)
            
            with open(train_file, 'a') as f:
                f.write(info1 + '\n')
                
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            
            # 调用修改后的val_epoch函数，返回多个指标
            val_metrics = val_epoch_brats21(
                model,
                val_loader,
                epoch=epoch,
                args=args,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            if args.rank == 0:
                # 提取指标
                dice_et = val_metrics['dice']['ET']
                dice_wt = val_metrics['dice']['WT']
                dice_tc = val_metrics['dice']['TC']
                dice_mean = val_metrics['dice']['mean']
                
                jaccard_et = val_metrics['jaccard']['ET']
                jaccard_wt = val_metrics['jaccard']['WT']
                jaccard_tc = val_metrics['jaccard']['TC']
                jaccard_mean = val_metrics['jaccard']['mean']
                
                hd95_et = val_metrics['hd95']['ET']
                hd95_wt = val_metrics['hd95']['WT']
                hd95_tc = val_metrics['hd95']['TC']
                hd95_mean = val_metrics['hd95']['mean']
                
                asd_et = val_metrics['asd']['ET']
                asd_wt = val_metrics['asd']['WT']
                asd_tc = val_metrics['asd']['TC']
                asd_mean = val_metrics['asd']['mean']
                
                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    "\nDice - ET: {:.4f}, WT: {:.4f}, TC: {:.4f}, Mean: {:.4f}".format(
                        dice_et, dice_wt, dice_tc, dice_mean),
                    "\nJaccard - ET: {:.4f}, WT: {:.4f}, TC: {:.4f}, Mean: {:.4f}".format(
                        jaccard_et, jaccard_wt, jaccard_tc, jaccard_mean),
                    "\nHD95 - ET: {:.4f}, WT: {:.4f}, TC: {:.4f}, Mean: {:.4f}".format(
                        hd95_et, hd95_wt, hd95_tc, hd95_mean),
                    "\nASD - ET: {:.4f}, WT: {:.4f}, TC: {:.4f}, Mean: {:.4f}".format(
                        asd_et, asd_wt, asd_tc, asd_mean),
                    "\ntime {:.2f}s".format(time.time() - epoch_time),
                )

                if writer is not None:
                    # 写入TensorBoard
                    writer.add_scalar("Val_Dice_ET", dice_et, epoch)
                    writer.add_scalar("Val_Dice_WT", dice_wt, epoch)
                    writer.add_scalar("Val_Dice_TC", dice_tc, epoch)
                    writer.add_scalar("Val_Dice_Mean", dice_mean, epoch)
                    
                    writer.add_scalar("Val_Jaccard_ET", jaccard_et, epoch)
                    writer.add_scalar("Val_Jaccard_WT", jaccard_wt, epoch)
                    writer.add_scalar("Val_Jaccard_TC", jaccard_tc, epoch)
                    writer.add_scalar("Val_Jaccard_Mean", jaccard_mean, epoch)
                    
                    writer.add_scalar("Val_HD95_ET", hd95_et, epoch)
                    writer.add_scalar("Val_HD95_WT", hd95_wt, epoch)
                    writer.add_scalar("Val_HD95_TC", hd95_tc, epoch)
                    writer.add_scalar("Val_HD95_Mean", hd95_mean, epoch)
                    
                    writer.add_scalar("Val_ASD_ET", asd_et, epoch)
                    writer.add_scalar("Val_ASD_WT", asd_wt, epoch)
                    writer.add_scalar("Val_ASD_TC", asd_tc, epoch)
                    writer.add_scalar("Val_ASD_Mean", asd_mean, epoch)
                    
                    # 写入日志文件
                    info2 = (
                        f"Final validation stats {epoch}/{args.max_epochs - 1}\n"
                        f"Dice - ET: {dice_et:.4f}, WT: {dice_wt:.4f}, TC: {dice_tc:.4f}, Mean: {dice_mean:.4f}\n"
                        f"Jaccard - ET: {jaccard_et:.4f}, WT: {jaccard_wt:.4f}, TC: {jaccard_tc:.4f}, Mean: {jaccard_mean:.4f}\n"
                        f"HD95 - ET: {hd95_et:.4f}, WT: {hd95_wt:.4f}, TC: {hd95_tc:.4f}, Mean: {hd95_mean:.4f}\n"
                        f"ASD - ET: {asd_et:.4f}, WT: {asd_wt:.4f}, TC: {asd_tc:.4f}, Mean: {asd_mean:.4f}\n"
                        f"time {time.time() - epoch_time:.2f}s"
                    )
                    
                    with open(train_file, 'a') as f:
                        f.write(info2 + '\n\n')
                
                # 使用平均Dice作为选择最佳模型的标准
                if dice_mean > best_metrics['dice']['mean']:
                    print(f"new best Dice ({best_metrics['dice']['mean']:.6f} --> {dice_mean:.6f}). ")
                    
                    # 更新所有最佳指标
                    best_metrics = val_metrics
                    
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=dice_mean, optimizer=optimizer, scheduler=scheduler
                        )
            
            # 记录最佳指标
            info4 = (
                f"Best metrics after epoch {epoch}:\n"
                f"Dice - ET: {best_metrics['dice']['ET']:.4f}, WT: {best_metrics['dice']['WT']:.4f}, "
                f"TC: {best_metrics['dice']['TC']:.4f}, Mean: {best_metrics['dice']['mean']:.4f}\n"
                f"Jaccard - ET: {best_metrics['jaccard']['ET']:.4f}, WT: {best_metrics['jaccard']['WT']:.4f}, "
                f"TC: {best_metrics['jaccard']['TC']:.4f}, Mean: {best_metrics['jaccard']['mean']:.4f}\n"
                f"HD95 - ET: {best_metrics['hd95']['ET']:.4f}, WT: {best_metrics['hd95']['WT']:.4f}, "
                f"TC: {best_metrics['hd95']['TC']:.4f}, Mean: {best_metrics['hd95']['mean']:.4f}\n"
                f"ASD - ET: {best_metrics['asd']['ET']:.4f}, WT: {best_metrics['asd']['WT']:.4f}, "
                f"TC: {best_metrics['asd']['TC']:.4f}, Mean: {best_metrics['asd']['mean']:.4f}"
            )
            
            with open(train_file, 'a') as f:
                f.write(info4 + '\n\n')
            
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=best_metrics['dice']['mean'], filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))   

        if scheduler is not None:
            scheduler.step()

    # 训练结束，打印最终最佳指标
    if args.rank == 0:
        print("\n" + "="*80)
        print("Training Finished!")
        print("Best Metrics:")
        print(f"Dice - ET: {best_metrics['dice']['ET']:.4f}, WT: {best_metrics['dice']['WT']:.4f}, "
              f"TC: {best_metrics['dice']['TC']:.4f}, Mean: {best_metrics['dice']['mean']:.4f}")
        print(f"Jaccard - ET: {best_metrics['jaccard']['ET']:.4f}, WT: {best_metrics['jaccard']['WT']:.4f}, "
              f"TC: {best_metrics['jaccard']['TC']:.4f}, Mean: {best_metrics['jaccard']['mean']:.4f}")
        print(f"HD95 - ET: {best_metrics['hd95']['ET']:.4f}, WT: {best_metrics['hd95']['WT']:.4f}, "
              f"TC: {best_metrics['hd95']['TC']:.4f}, Mean: {best_metrics['hd95']['mean']:.4f}")
        print(f"ASD - ET: {best_metrics['asd']['ET']:.4f}, WT: {best_metrics['asd']['WT']:.4f}, "
              f"TC: {best_metrics['asd']['TC']:.4f}, Mean: {best_metrics['asd']['mean']:.4f}")
        print("="*80)
    
    return best_metrics['dice']['mean']




def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)