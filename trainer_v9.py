import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import time
import os
import shutil
import warnings
import copy
from collections import deque
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.utils import first
from monai.transforms import AsDiscrete, Activations
from utils.utils import *

# 添加必要的辅助函数和类
def distributed_all_gather(tensor_list, out_numpy=True, is_valid=False):
    """简易的分布式聚合实现"""
    gathered_list = []
    for tensor in tensor_list:
        gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_t, tensor)
        gathered_list.append(torch.cat(gather_t, dim=0))
    return gathered_list

class AverageMeter:
    """计算并存储平均值和当前值"""
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
        self.avg = self.sum / self.count if self.count != 0 else 0

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



class MedicalRewardEvaluator:
    """医疗奖励评估器 - 加入解剖约束"""
    def __init__(self, organ_type='left_atrium'):
        self.organ_type = organ_type
    
    def compute_reward(self, seg_pred, target, seg_pred_ori):
        # 基础Dice分数 (分割准确性)
        dice1 = (2 * (seg_pred.detach() * target).sum()) / (seg_pred.detach().sum() + target.sum() + 1e-5)
        dice2 = (2 * (seg_pred_ori.detach() * target).sum()) / (seg_pred_ori.detach().sum() + target.sum() + 1e-5)
        
        dice = dice1 - dice2
        
        # 结构连续性奖励 (防止碎片化)
        continuity = self.compute_structural_continuity(seg_pred)
        
        return dice + 0.3 * continuity
    
    def compute_structural_continuity(self, pred_mask):
        """计算器官结构连续性得分"""
        # 计算连通域数量 (越少越好)
        num_components = self._count_connected_components(pred_mask)
        return 1.0 - min(0.3, num_components * 0.05)  # 最多扣0.3分
    
    def _count_connected_components(self, mask):
        """计算连通域数量 (简化版)"""
        B = mask.shape[0]
        counts = []
        for b in range(B):
            non_zero = (mask[b] > 0.5).sum()
            # 简化的连通域估算
            counts.append(1 if non_zero > 10 else 0)  # 实际应用中应使用连通域分析
        return max(counts)  # 取最大连通域数

class ParameterServer:
    def __init__(self, model, optimizer=None):
        self.global_state = {
            #'model': copy.deepcopy(model).cpu().state_dict(),
            'model': copy.deepcopy(model).state_dict(),
            'optim': copy.deepcopy(optimizer).state_dict() if optimizer else None
        }
        self.local_model = model.cuda()
        self.local_optim = optimizer

    def sync_parameters(self, direction='global_to_local', include_optim=False):
        if direction == 'global_to_local':
            self.local_model.load_state_dict(self.global_state['model'])
            if include_optim and self.local_optim and self.global_state['optim']:
                self.local_optim.load_state_dict(self.global_state['optim'])
        else:
            self.global_state['model'] = copy.deepcopy(self.local_model.state_dict())
            if include_optim and self.local_optim:
                self.global_state['optim'] = copy.deepcopy(self.local_optim.state_dict())

class MedicalImageTransitionBuffer:
    def __init__(self, capacity, num_actions, model, temperature=0.8):
        self.buffer = deque(maxlen=capacity)
        self.num_actions = num_actions  # 现在是3个动作：0=无操作, 1=增强, 2=减弱
        self.temperature = temperature
        self.model = model
        self.reward_evaluator = MedicalRewardEvaluator()
        
    def _select_actions(self, action_probs, target_shape=None):
        B, C, D, H, W = action_probs.shape
        
        # 添加维度调整 - 确保与输入空间维度一致
        if target_shape and (D, H, W) != target_shape:
            action_probs = F.interpolate(
                action_probs, 
                size=target_shape, 
                mode='trilinear',
                align_corners=False
            )
            D, H, W = target_shape
        
        adjusted_probs = F.softmax(action_probs / self.temperature, dim=1)
        actions = torch.multinomial(
            adjusted_probs.permute(0, 2, 3, 4, 1).reshape(-1, C), 
            1
        ).reshape(B, D, H, W)
        return actions
    
    def _transform_labels(self, label, actions, image_shape):
        # 现在只有3个动作（0,1,2），不再需要复杂的标签变换
        # 因为我们移除了会改变标签形态的弹性变形和结构变化
        return label.clone()  # 直接返回原标签

    def _apply_actions_to_image(self, image, actions):
        next_image = image.clone()
        B, _, D, H, W = image.shape
        
        for action_type in range(self.num_actions):
            action_mask = (actions == action_type)
            if action_mask.any():
                next_image = self._apply_medical_effect(next_image, action_mask, action_type)
            
        return next_image
    
    def _apply_medical_effect(self, image, action_mask, action_type):
        modified_image = image.clone()
        B, _, D, H, W = image.shape
        
        for b in range(B):
            current_mask = action_mask[b]
            if not current_mask.any():
                continue
                
            current_img = image[b, 0]
            intensities = current_img[current_mask]
            
            # 动作类型处理...
            if action_type == 0:  # 无操作
                pass
                
            elif action_type == 1:  # 增强病灶/组织
                enhancement = 1.0 + torch.rand_like(intensities) * 0.3
                modified_intensities = torch.clamp(intensities * enhancement, 0.0, 1.0)
                modified_image[b, 0][current_mask] = modified_intensities
                
            elif action_type == 2:  # 减弱病灶/组织
                reduction = 1.0 - torch.rand_like(intensities) * 0.3
                modified_intensities = torch.clamp(intensities * reduction, 0.0, 1.0)
                modified_image[b, 0][current_mask] = modified_intensities
                
        return modified_image
    
    def push(self, image, label, action_probs):
        # 获取图像维度作为目标形状
        B, _, D_img, H_img, W_img = image.shape
        target_shape = (D_img, H_img, W_img)
        
        # 1. 从动作概率采样具体动作
        actions = self._select_actions(action_probs, target_shape)
        
        # 2. 根据动作修改图像状态
        next_image = self._apply_actions_to_image(image, actions)
        
        # 3. 标签不需要变换
        next_label = label.clone()
        
        # 4. 计算奖励
        with torch.no_grad():
            seg_pred_ori, _, _ = self.model(image)
            seg_pred, _, _ = self.model(next_image)
            reward = self.reward_evaluator.compute_reward(seg_pred, label, seg_pred_ori)
        
        # 5. 存储到缓冲区
        self.buffer.append({
            'image': image.detach().cpu(),
            'label': label.detach().cpu(),
            'action_probs': action_probs.detach().cpu(),
            'reward': torch.tensor([reward] * image.shape[0]),
            'next_image': next_image.detach().cpu(),
            'next_label': next_label.detach().cpu(),
            'selected_actions': actions.detach().cpu(),
            'original_shape': (D_img, H_img, W_img)
        })
    
    def get_sequence(self):
        """获取完整图像状态轨迹"""
        return list(self.buffer)

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    segment_loss = AverageMeter()
    p_loss = AverageMeter()
    v_loss = AverageMeter()
    
    # 多阶段训练策略
    args.warmup_epochs = getattr(args, 'warmup_epochs', 200)
    args.rl_start_epoch = getattr(args, 'rl_start_epoch', 300)
    param_server = ParameterServer(model, optimizer)
    
    # 确定当前训练阶段
    if epoch < args.warmup_epochs:
        training_phase = 'warmup'
    elif epoch < args.rl_start_epoch:
        training_phase = 'transition'
    else:
        training_phase = 'full_rl'
    
    for idx, batch_data in enumerate(loader):
    
        '''
        if idx >= 0:
           break
        ''' 
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        
        # ========== 仅用于分割训练的路径 (warmup阶段) ==========
        if training_phase == 'warmup':
            optimizer.zero_grad()
            outputs = model(data)
            
            # 处理不同格式的输出
            if isinstance(outputs, tuple):
                # 如果是元组，取第一个元素作为分割预测
                seg_pred = outputs[0]
            else:
                seg_pred = outputs
            
            # 计算分割损失
            seg_loss = loss_func(seg_pred, target)
            total_loss = seg_loss
            
            # 反向传播和参数更新
            if args.amp:
                scaler.scale(total_loss).backward()
                
                # 分布式梯度聚合 (只在分布式训练中执行)
                if args.distributed:
                    for param in model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= dist.get_world_size()
                
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                
                # 分布式梯度聚合 (只在分布式训练中执行)
                if args.distributed:
                    for param in model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= dist.get_world_size()
                
                optimizer.step()
            
            # 5. 监控和日志记录
            run_loss.update(total_loss.item(), n=data.size(0))
            segment_loss.update(seg_loss.item(), n=args.batch_size)
            
            if args.rank == 0:
                print(
                    "Warmup Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "seg_loss: {:.4f}".format(segment_loss.avg),
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
            
            # 重要修改：在warmup阶段后立即同步参数（确保参数更新到全局）
            #param_server = ParameterServer(model, optimizer)
            param_server.sync_parameters('local_to_global', include_optim=True)
            continue  # 跳过强化学习部分
        
        # ============= 强化学习相关部分 (transition和full_rl阶段) =============
        # 1. 初始化参数服务器
        '''
        param_server = ParameterServer(model, optimizer)
        param_server.sync_parameters('global_to_local', include_optim=True)
        '''
        
        # 2. 创建状态转移缓冲区
        buffer = MedicalImageTransitionBuffer(
            args.t_max, 3, param_server.local_model
        )
        
        # 3. 保存原始空间维度
        original_shape = data.shape[2:]
        
        # 4. 初始化当前数据（图像状态）
        current_data = data.clone()
        current_label = target.clone()
        
        optimizer.zero_grad()
        
        # 5. 多步轨迹采集
        for t in range(args.t_start, args.t_max):
            with torch.enable_grad():
                # 获取模型输出（分割预测、动作概率、状态价值）
                seg_pred, action_probs, state_value = param_server.local_model(current_data)
                
                # 存储状态转移
                buffer.push(current_data, current_label, action_probs)
                
                # 更新状态和标签
                transition = buffer.buffer[-1]
                current_data = transition['next_image'].cuda(args.rank)
                current_label = transition['next_label'].cuda(args.rank)
                
                '''
                # 确保数据维度一致
                if current_data.shape[2:] != original_shape:
                    current_data = F.interpolate(
                        current_data, 
                        size=original_shape, 
                        mode='trilinear',
                        align_corners=False
                    )
                
                if current_label.shape[2:] != original_shape:
                    current_label = F.interpolate(
                        current_label.float(), 
                        size=original_shape, 
                        mode='nearest'
                    )
                '''
                
        # 6. 获取所有状态转移
        transitions = buffer.get_sequence()
        if len(transitions) < 1:
            continue
            
        # 7. 计算回报
        returns = torch.zeros(len(transitions), device=data.device)
        
        # 终止状态价值
        last_state = transitions[-1]['next_image'].cuda()
        with torch.no_grad():
            _, _, last_value = param_server.local_model(last_state)
        returns[-1] = last_value.squeeze()
        
        # 计算折扣回报
        for t in reversed(range(len(transitions)-1)):
            returns[t] = transitions[t]['reward'].cuda() + args.gamma * returns[t+1]
        
        # 8. 梯度累积
        policy_grad = 0
        value_grad = 0
        seg_grad = 0
        
        for t in range(len(transitions)):
            state = transitions[t]['image'].cuda()
            target_shape = transitions[t]['original_shape']
            
            curr_label = transitions[t]['label'].cuda().float()
            
            # 前向计算
            seg_pred, action_probs, state_value = param_server.local_model(state)
            
            # 获取实际选择的动作
            action = transitions[t]['selected_actions'].cuda().long()
            
            # 策略梯度计算
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)))
            advantage = (returns[t] - state_value.detach()).squeeze()
            policy_grad += -(log_prob * advantage).mean()
            
            # 价值梯度计算
            value_grad += 0.5 * (returns[t] - state_value).pow(2).mean()
            
            '''
            # 分割梯度计算 (确保维度匹配)
            B, C, D, H, W = curr_label.shape
            if seg_pred.shape[2:] != (D, H, W):
                seg_pred = F.interpolate(
                    seg_pred, 
                    size=(D, H, W),
                    mode='trilinear',
                    align_corners=False
                )
            '''
            seg_grad += loss_func(seg_pred, curr_label)
        
        # 9. 平均各时间步的梯度
        num_transitions = max(1, len(transitions))
        policy_grad /= num_transitions
        value_grad /= num_transitions
        seg_grad /= num_transitions
        
        # 10. 多阶段训练策略组合损失
        if training_phase == 'transition':
            total_loss = 0.7 * seg_grad + 0.3 * value_grad
        else:  # full_rl
            total_loss = 0.6 * seg_grad + 0.2 * policy_grad + 0.2 * value_grad
        
        # 11. 反向传播和参数更新
        if args.amp:
            scaler.scale(total_loss).backward()
            # 分布式梯度聚合
            if args.distributed:
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= dist.get_world_size()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            # 分布式梯度聚合
            if args.distributed:
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= dist.get_world_size()
            optimizer.step()
        
        # 12. 参数同步
        param_server.sync_parameters('local_to_global', include_optim=True)
        
        # 13. 监控和日志记录
        run_loss.update(total_loss.item(), n=data.size(0))
        segment_loss.update(seg_grad.item(), n=args.batch_size)
        if training_phase == 'full_rl':
            p_loss.update(policy_grad.item(), n=args.batch_size)
        if training_phase != 'warmup':
            v_loss.update(value_grad.item(), n=args.batch_size)
        
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "total_loss: {:.4f}".format(run_loss.avg),
                "seg_loss: {:.4f}".format(segment_loss.avg),
                ("policy_loss: {:.4f}".format(p_loss.avg) if training_phase == 'full_rl' else ""),
                ("value_loss: {:.4f}".format(v_loss.avg) if training_phase != 'warmup' else ""),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    
    # 清理梯度
    for param in model.parameters():
        param.grad = None
        
    # 根据阶段返回适当的值
    if training_phase == 'warmup':
        return run_loss.avg, segment_loss.avg, None, None
    else:
        return run_loss.avg, segment_loss.avg, p_loss.avg, v_loss.avg


def val_epoch(param_server, loader, epoch, acc_func, args, 
             model_inferer=None, post_sigmoid=None, post_pred=None):
    
    '''
    # 保存原始全局状态
    original_state = {
        'model': copy.deepcopy(param_server.global_state['model']),
        'optim': copy.deepcopy(param_server.global_state['optim'])
    }
    '''
    
    # 获取模型实例并加载全局参数
    model = param_server.local_model
    model.load_state_dict(param_server.global_state['model'])
    
    # 设备切换与模型模式设置
    
    model = model.to(args.rank)
    model.eval()
    
    run_acc = AverageMeter()
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
                # 使用模型推理器获取预测结果
                #seg_pred, _, _ = model(data)
                seg_pred = model_inferer(data)
                logits = seg_pred
            
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
            
            
                
    '''
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
                Dice = run_acc.avg
                
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    ", Dice:",
                    Dice,
                    ", time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    
    return run_acc.avg
  '''
  
  
# 主训练循环
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
    # 参数初始化
    writer = None
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
    param_server = ParameterServer(model, optimizer)
    
    # 添加多阶段训练参数
    args.warmup_epochs = getattr(args, 'warmup_epochs', 200)
    args.rl_start_epoch = getattr(args, 'rl_start_epoch', 300)
    
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        
        # 训练阶段
        total_loss, seg_loss, p_loss, v_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(total_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", total_loss, epoch)
            writer.add_scalar("train_seg_loss", seg_loss, epoch)
            if p_loss is not None:
                writer.add_scalar("train_policy_loss", p_loss, epoch)
            if v_loss is not None:
                writer.add_scalar("train_value_loss", v_loss, epoch)
                
            info1 = "epoch:{}, train_loss:{:.4f}, seg_loss:{:.4f}, policy_loss:{:.4f}, value_loss:{:.4f}".format(
                epoch, total_loss, seg_loss, p_loss if p_loss is not None else 0, v_loss if v_loss is not None else 0
            )
            with open(train_file, 'a') as f:
                f.write(info1 + '\n')
                
        # 验证阶段
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            
            # param_server = ParameterServer(model, optimizer)
            # 参数同步
            param_server.sync_parameters('local_to_global', include_optim=True)
            
            '''
            val_acc = val_epoch(
                param_server,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            '''
            
            '''
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
            '''
            # 调用修改后的val_epoch函数
            dice_avg, jaccard_avg, hd95_avg, asd_avg = val_epoch(
                param_server,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
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
    '''
            if args.rank == 0:
                Dice = val_acc[0] if isinstance(val_acc, (tuple, list)) and len(val_acc) > 0 else val_acc
                Dice = float(Dice) if isinstance(Dice, torch.Tensor) else Dice
                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    ", Dice:", Dice,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )

                if writer is not None:
                    writer.add_scalar("Val_Mean_Dice", Dice, epoch)
                    
                    info2 = "Val epoch {}: Dice={:.4f}".format(epoch, Dice[0])
                    
                    with open(train_file, 'a') as f:
                        f.write(info2 + '\n')
                    
                    # 添加类别指标
                    if semantic_classes is not None and isinstance(val_acc, (tuple, list)):
                        for i, cls_name in enumerate(semantic_classes):
                            if i < len(val_acc):
                                val_acc_i = val_acc[i] if isinstance(val_acc[i], (int, float)) else val_acc[i].item()
                                writer.add_scalar("Val_{}".format(cls_name), val_acc_i, epoch)
                
                
                combined_score = Dice[0]
                if combined_score > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, combined_score))
                    val_acc_max = combined_score
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            
            if args.rank == 0:
                info4 = "best_dice_Score:{:.6f}".format(val_acc_max)            
                with open(train_file, 'a') as f:
                    f.write(info4 + '\n')
            
            # 定期保存最终模型
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(
                        os.path.join(args.logdir, "model_final.pt"), 
                        os.path.join(args.logdir, "model.pt")
                    )

        # 学习率调整
        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Combined Score: ", val_acc_max)
    return val_acc_max
    '''