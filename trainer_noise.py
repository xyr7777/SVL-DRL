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

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from monai.data import decollate_batch
from policy_value import PixelDRL_MG_MultiClass
#from policy_value_1 import PixelDRL_MG_MultiClass


def train_epoch(value_net, policy_net, loader, optimizer, scaler, epoch, loss_func, args):
    
    policy_net.train()
    # 训练记录
    history = {
        'policy_loss': [],
        'value_loss': [],
        'dice': []
    }
    model = PixelDRL_MG_MultiClass(num_classes = args.out_channels, policy_net=policy_net, value_net=value_net)

    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        
        '''
        if idx > 10:
            break
        '''
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        
        for param in policy_net.parameters():
            param.grad = None
        for param in value_net.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            losses = model.train_step(data, target)
        
        # 记录指标
        history['policy_loss'].append(losses['policy_loss'])
        history['value_loss'].append(losses['value_loss'])
        history['dice'].append(losses['final_dice'])
        
        print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "policy_loss: {:.4f}".format(losses['policy_loss']),
                "value_loss: {:.4f}".format(losses['policy_loss']),
                "dice: {:.4f}".format(losses['final_dice']),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()    
    
    history['policy_loss'] = sum(history['policy_loss']) / len(history['policy_loss'])
    history['value_loss'] = sum(history['value_loss']) / len(history['value_loss'])
    history['dice'] = sum(history['dice']) / len(history['dice'])
    
    return history


def val_epoch(value_net, policy_net, loader, epoch, acc_func, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    start_time = time.time()
    run_acc = AverageMeter()
    model = PixelDRL_MG_MultiClass(num_classes = args.out_channels, policy_net=policy_net, value_net=value_net)
    policy_net.eval()
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            '''
            if idx > 10:
                break
            '''
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            
            
            with autocast(enabled=args.amp):
                #logits = model_inferer(data)
                logits = model._rollout(data)
                     
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
                Dice_1 = run_acc.avg[0]
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    ", dice:",
                    Dice_1,
                    ", time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()

    return run_acc.avg


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


def run_training_noise(
    value_net,
    policy_net,
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
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    model = PixelDRL_MG_MultiClass(num_classes = args.out_channels, policy_net=policy_net, value_net=value_net)
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
        
        train_result = train_epoch(
            value_net, policy_net, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "policy_loss: {:.4f}".format(train_result['policy_loss']),
                "value_loss: {:.4f}".format(train_result['value_loss']),
                "dice: {:.4f}".format(train_result['dice']),
                
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            #writer.add_scalar("train_loss", train_loss, epoch)
            info1 = ("epoch:{} ,dice:{}".format(epoch, train_result['dice']))
            
            with open(train_file,'a') as f:
                f.write(info1 + '\n')
                
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc = val_epoch(
                value_net, 
                policy_net, 
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            if args.rank == 0:
                Dice_1 = val_acc[0]
                
                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    "dice:",
                    Dice_1,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )

                if writer is not None:
                    writer.add_scalar("Mean_Val_Dice", np.mean(val_acc), epoch)
                    
                    info2 = (
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1)+
                    ", dice:{:.2f}".format(
                    Dice_1) +
                    ", time {:.2f}s".format(time.time() - epoch_time)
                )
                    info3 = ("epoch:{} Mean_Val_Dice:{}".format(epoch, np.mean(val_acc)))
                    with open(train_file,'a') as f:
                        f.write(info2 + '\n' + info3 + '\n')
                    if semantic_classes is not None:
                        for val_channel_ind in range(len(semantic_classes)):
                            if val_channel_ind < val_acc.size:
                                writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
                val_avg_acc = np.mean(val_acc)
                '''
                with open(train_file,'a') as f:
                        f.write('val_acc_best:{}'.format(val_acc_max))
                '''
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        model.save_checkpoint(args.logdir+'/model_final.pt', epoch)
            info4 = ("best_Val_Dice:{}".format(val_acc_max))            
            with open(train_file,'a') as f:
                f.write(info4 + '\n')
            
                
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                #save_checkpoint(policy_net, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                model.save_checkpoint(args.logdir + "/model.pt", args.max_epochs)
                '''
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))   
                '''

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max




class TrainingFramework:
    def __init__(self, device, num_classes=3):
        self.device = device
        self.num_classes = num_classes
        
        # 初始化网络
        self.policy_net = UNet(
            spatial_dims=3,
            in_channels=3 * 2,  # 原始图像(3) + 当前预测(3)
            out_channels=3,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2
        ).to(device)
        
        self.value_net = UNet(
            spatial_dims=3,
            in_channels=3 * 2,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2
        ).to(device)
        
        # 初始化训练器
        self.trainer = PixelDRL_MG_MultiClass(
            num_classes=num_classes,
            policy_net=self.policy_net,
            value_net=self.value_net,
            device=device
        )

    def generate_batch(self, batch_size=1):
        """生成随机3D数据批次"""
        depth = height = width = 96
        
        # 生成随机图像和标签
        X = torch.randn(batch_size, 3, depth, height, width).to(self.device)
        
        # 生成随机分割标签
        G = torch.randint(0, self.num_classes, 
                        (batch_size, 1, depth, height, width)).long().to(self.device)
        
        # 转换为one-hot编码
        G_onehot = torch.zeros(batch_size, self.num_classes, 
                             depth, height, width).to(self.device)
        G_onehot.scatter_(1, G, 1.0)
        
        return X, G_onehot

    def train(self, num_epochs=100, batch_size=2):
        """完整训练流程"""
        # 训练记录
        history = {
            'policy_loss': [],
            'value_loss': [],
            'dice': []
        }
        
        # 训练循环
        for epoch in range(num_epochs):
            # 生成训练数据
            X, G = self.generate_batch(batch_size)
            
            # 执行训练步骤
            losses = self.trainer.train_step(X, G)
            
            # 记录指标
            history['policy_loss'].append(losses['policy_loss'])
            history['value_loss'].append(losses['value_loss'])
            history['dice'].append(losses['final_dice'])
            
            # 打印进度
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                    f"Policy Loss: {losses['policy_loss']:.4f}, "
                    f"Value Loss: {losses['value_loss']:.4f}, "
                    f"Dice: {losses['final_dice']:.4f}")
        
        # 训练结束保存模型
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
        }, 'final_model.pth')
        
        return history

if __name__ == "__main__":
    # 配置训练环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化训练框架
    framework = TrainingFramework(device=device, num_classes=3)
    
    # 启动训练
    training_history = framework.train(
        num_epochs=100,
        batch_size=2
    )




















