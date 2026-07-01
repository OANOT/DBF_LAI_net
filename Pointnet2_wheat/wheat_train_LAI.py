"""
Author: Benny & Optimized for Multi-Seed LAI Analysis with Auto GPU Selection
"""
import argparse
import os
import subprocess
import numpy as np
import io
import sys


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# --- 1. 自动检测空闲 GPU 的函数 ---
def get_free_gpu():
    """通过 nvidia-smi 查询显存占用，返回最空闲 GPU 的编号"""
    try:
        # 执行 nvidia-smi 命令查询显存使用情况
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader"
        output = subprocess.check_output(cmd.split())
        # 解析输出，转换为整数列表 [used_mem_gpu0, used_mem_gpu1, ...]
        memory_used = [int(x) for x in output.decode('utf-8').strip().split('\n')]
        # 返回显存占用最小的 GPU 索引
        best_gpu = str(np.argmin(memory_used))
        # print(f"📡 自动检测完成：当前最空闲的 GPU 为 Index {best_gpu} (显存占用: {memory_used[int(best_gpu)]} MiB)")
        return best_gpu
    except Exception as e:
        print(f"⚠️ 自动检测 GPU 失败 ({e})，默认使用 GPU 0")
        return "0"

# --- 2. 核心：在任何 torch 调用之前设置环境变量 ---
def setup_gpu_environment(gpu_arg):
    if gpu_arg == 'auto':
        target_gpu = get_free_gpu()
    else:
        target_gpu = gpu_arg
    
    os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu
    # print(f"✅ 已锁定运行环境：CUDA_VISIBLE_DEVICES = {target_gpu}")

# 先解析一次命令行，看用户是否指定了特定 GPU 或 auto
def pre_parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu', type=str, default='auto') # 默认为 auto
    args, _ = parser.parse_known_args()
    return args

# 立即执行环境设置
pre_args = pre_parse_args()
setup_gpu_environment(pre_args.gpu)

# --- 3. 现在可以安全地导入 torch 和其他库了 ---
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import time
import torch.nn as nn
from utils import draw_scatter

import pandas as pd
import random
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# 确保矩阵运算确定性
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default=['DBF_LAI'], help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--epoch', default=200, type=int, help='Epoch to run')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use: e.g. "0", "1", or "auto"')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default='log', help='Log path')
    parser.add_argument('--decay_rate', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=40, help='Early stopping')
    parser.add_argument('--npoint', type=int, default=16384, help='Point Number')
    parser.add_argument('--seeds', type=int, nargs='+', default=[123,23], help='List of seeds')
    parser.add_argument('--types', type=str, default=['base'], help='Fusion tyspe') # 'base', 'transfer'
    parser.add_argument('--point_path', type=str, default='data/npy_data_LAI')
    parser.add_argument('--ms_path', type=str, default='data/ms_npy_LAI')
    parser.add_argument('--stats_path', type=str, default='LAI_data.csv')

    return parser.parse_args()

def run_train_eval(model, seed, type, args, global_log_string):
    # 每个子实验开始前固定随机状态
    seed_everything(seed)
    if type == 'base':
        from data_utils.LAIDataLoader import WheatLAIDataset
    elif type == 'transfer':
        from data_utils.LAIDataLoader_transfer import WheatLAIDataset
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('Pointnet2_wheat').joinpath(args.log_dir).joinpath(model).joinpath(type).joinpath(f'LAI_{seed}_{timestr}') 
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(f"Model_Seed_{seed}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % experiment_dir, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    def log_string(str):
        logger.info(str)
        print(str)

    log_string(f'🚀 Starting Experiment with Seed: {seed}')

    '''DATA LOADING'''
    DATA_PATH = args.point_path
    MS_PATH = args.ms_path
    TRAIN_DATASET = WheatLAIDataset(csv_path=args.stats_path, data_root=DATA_PATH, split='train', ms_root=MS_PATH, num_point=args.npoint, splitSeed=seed)
    VAL_DATASET   = WheatLAIDataset(csv_path=args.stats_path, data_root=DATA_PATH, split='val',  ms_root=MS_PATH,  num_point=args.npoint, splitSeed=seed)
    TEST_DATASET  = WheatLAIDataset(csv_path=args.stats_path, data_root=DATA_PATH, split='test',  ms_root=MS_PATH, num_point=args.npoint, splitSeed=seed)
    
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    valDataLoader   = torch.utils.data.DataLoader(VAL_DATASET,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    testDataLoader  = torch.utils.data.DataLoader(TEST_DATASET,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    '''MODEL LOADING'''
    # MODEL = importlib.import_module(args.model)
    MODEL = importlib.import_module(model)
    classifier = MODEL.get_model(num_class=1, stats_dim=len(TRAIN_DATASET.stats_cols)).cuda()
    criterion = nn.SmoothL1Loss().cuda()
    classifier.apply(inplace_relu)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)

    best_val_metric = float('inf')

    # 设置早停参数
    early_stopping_patience = args.early_stopping_patience
    early_stopping_counter = 0  # 跟踪验证损失是否改善的计数器
    # --- 新增：保存模型结构和超参数 ---
    info_savepath = str(experiment_dir.joinpath(f'model_structure.txt'))
    with open(info_savepath, 'w', encoding='utf-8') as f:
        # 1. 写入超参数 (args 对象通常包含所有配置)
        f.write("="*20 + " Hyperparameters " + "="*20 + "\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg:<30}: {value}\n")
        
        f.write("\n" + "="*20 + " Model Architecture " + "="*20 + "\n")
        # 2. 写入模型整体结构字符串
        f.write(str(classifier) + "\n\n")

        f.write("="*20 + " Detailed Layer Definitions " + "="*20 + "\n")
        # 3. 写入每一层的具体参数形状 (这反映了内部结构)
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                f.write(f"{name:<40} | Shape: {str(list(param.shape)):<20}\n")
                
    log_string(f'📝 模型结构与超参数已保存到: {info_savepath}')
    # -----------------------------------------
    for epoch in range(args.epoch):
        log_string(f'--- Seed {seed} | Epoch {epoch+1}/{args.epoch} ---')
        classifier.train()
        train_pred, train_gt = [], []
        
        for i, (points, stats, ms_images, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points, stats, ms_images, target = points.cuda(), stats.cuda(), ms_images.cuda(), target.cuda()
            pred, _ = classifier(points, stats, ms_images)
            loss = criterion(pred.view(-1), target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            train_pred.extend(pred.detach().cpu().numpy().flatten())
            train_gt.extend(target.cpu().numpy().flatten())
        
        train_pred = np.array(train_pred)
        train_gt = np.array(train_gt)
        train_rmse = np.sqrt(mean_squared_error(train_gt, train_pred))
        train_r2 = r2_score(train_gt, train_pred)
        log_string(f'Train R^2: {train_r2:.4f}')
        log_string(f'Train RMSE: {train_rmse:.4f}\n')

        # valid_loss = 0.0
        with torch.no_grad():
            classifier.eval()
            val_pred, val_gt = [], []
            # for i, (points, stats, ms_images, target) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
            for points, stats, ms_images, target in valDataLoader:
                points, stats, ms_images, target = points.cuda(), stats.cuda(), ms_images.cuda(), target.cuda()
                pred, _ = classifier(points, stats, ms_images)
                val_pred.extend(pred.cpu().numpy().flatten())
                val_gt.extend(target.cpu().numpy().flatten())
                # loss = criterion(pred, target)
                # valid_loss += loss.item() * points.size(0)

            val_pred = np.array(val_pred)
            val_gt = np.array(val_gt)
            val_rmse = np.sqrt(mean_squared_error(val_gt, val_pred))
            val_r2 = r2_score(val_gt, val_pred)
            log_string(f'Val R^2: {val_r2:.4f}')
            log_string(f'Val RMSE: {val_rmse:.4f}\n')


            current_metric = val_rmse
            if current_metric < best_val_metric:
                if epoch >= 0:
                    test_pred, test_gt = [], []
                    for points, stats, ms_images, target in testDataLoader:
                        points, stats, ms_images, target = points.cuda(), stats.cuda(), ms_images.cuda(), target.cuda()
                        pred, _ = classifier(points, stats, ms_images)
                        test_pred.extend(pred.cpu().numpy().flatten())
                        test_gt.extend(target.cpu().numpy().flatten())
                    
                    test_pred = np.array(test_pred)
                    test_gt = np.array(test_gt)
                    test_rmse = np.sqrt(mean_squared_error(test_gt, test_pred))
                    test_r2 = r2_score(test_gt, test_pred)
                    log_string(f'Test R^2: {test_r2:.4f}')
                    log_string(f'Test RMSE: {test_rmse:.4f}')
                    save_name = f'best_val_R2_{val_r2:.3f}_val_rmse_{val_rmse:.3f}_test_R2_{test_r2:.3f}_test_rmse_{test_rmse:.3f}_epoch_{epoch+1}.pth'
                    savepath = str(checkpoints_dir.joinpath(save_name))
                    torch.save(classifier.state_dict(), savepath)

                    # pd.DataFrame(test_pred).to_csv('pred.csv', index=False)
                    # pd.DataFrame(test_gt).to_csv('gt.csv', index=False)
                    log_string(f'🌟 New Best! Saved: {save_name}')
                best_val_metric = current_metric
                early_stopping_counter = 0                

            else:
                early_stopping_counter += 1
                log_string(f'No improvement. Counter: {early_stopping_counter}/{early_stopping_patience}')
                
                if early_stopping_counter >= early_stopping_patience:
                    log_string('🛑 Early Stopping triggered.')
                    break

        scheduler.step()
    logger.handlers.clear()

def main():         
    args = parse_args()
    print(f"模型列表: {args.model}")
    print(f"种子列表: {args.seeds}")
    for type in args.types:
        try:
            for model in args.model:
                try:
                    for seed in args.seeds:
                        try:
                            run_train_eval(model, seed, type, args, print)
                        except Exception as e:
                            print(f"种子 {seed} 运行失败: {e}")
                            continue
                except Exception as e:
                    print(f"模型 {seed} 运行失败: {e}")
                    continue
        except Exception as e:
            print(f"{model} 运行失败: {e}")
            continue


if __name__ == '__main__':
    main()
    # args = parse_args()
    # run_train_eval(args.seeds[0], args, print)
