import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime

class WheatLAIDataset(Dataset):
    def __init__(self, csv_path, data_root,ms_root, num_point=16384, 
                 split='train', split_log_dir='split_logs', splitSeed = 42):
        """
        加载预处理为 .npy 的 LAI 数据集。
        实现 8:1:1 划分：Train (80%), Val (10%), Test (10%)。
        
        参数:
        - split: 'train', 'val' 或 'test'
        """
        self.data_root = data_root
        self.num_point = num_point
        self.split = split
        self.df = pd.read_csv(csv_path)
        self.ms_root = ms_root

        self.df['physical_plot_id'] = self.df['plot_id'].apply(
            lambda x: str(x).split('_')[1] if '_' in str(x) else str(x)
        )
        unique_plots = self.df['physical_plot_id'].unique()
        
        # --- 2. 针对地块 ID 进行 8:1:1 随机切分 ---
        # 使用固定种子 42，确保多次运行划分结果完全一致
        np.random.seed(splitSeed) 
        np.random.shuffle(unique_plots)
        
        num_plots = len(unique_plots)
        train_end = int(0.7 * num_plots)
        val_end = int(0.85 * num_plots) # 80% + 10% = 90%
        
        train_plots = unique_plots[:train_end]
        val_plots = unique_plots[train_end:val_end]
        test_plots = unique_plots[val_end:]
        
        # --- 3. 记录切分日志 (仅在初始化训练集时执行一次) ---
        if split == 'train':
            os.makedirs(split_log_dir, exist_ok=True)
            log_list = []
            for p in train_plots: log_list.append({'physical_plot_id': p, 'set': 'train'})
            for p in val_plots:   log_list.append({'physical_plot_id': p, 'set': 'val'})
            for p in test_plots:  log_list.append({'physical_plot_id': p, 'set': 'test'})
            
            log_df = pd.DataFrame(log_list)
            log_path = os.path.join(split_log_dir, f'LAI_plot_level_811_record_{splitSeed}.csv')
            log_df.to_csv(log_path, index=False)
            print(f"✔️ 8:1:1 地块级切分记录已保存: {log_path}")

        # --- 4. 根据 split 筛选对应的行索引 ---
        if split == 'train':
            target_plots = train_plots
        elif split == 'val':
            target_plots = val_plots
        elif split == 'test':
            target_plots = test_plots
        else:
            raise ValueError("split 参数必须是 'train', 'val' 或 'test'")
            
        self.valid_indices = self.df[self.df['physical_plot_id'].isin(target_plots)].index.values


        metric_names = ['PH_max', 'PH_mean', 'PH_sd', 'PH_skew', 'PH_kurt'] + [f'PH_q{i}' for i in range(5, 100, 5)] \
                    + [f'PH_pcum{i}' for i in range(1, 10)]
        # self.stats_cols = [c for c in self.df.columns if c.startswith("PH_")]  + ["n"]
        self.stats_cols = metric_names
        # 💡 关键：无论当前是哪个 split，均值和标准差都必须从【训练地块】中计算
        train_df_stats = self.df[self.df['physical_plot_id'].isin(train_plots)][self.stats_cols]
        self.stats_mean = train_df_stats.mean().values
        self.stats_std = train_df_stats.std().values + 1e-8 # 加微小值防止除以0
        
        print(f"已加载 {split} 数据集: {len(self.valid_indices)} 个样本 "
              f"(涉及 {len(target_plots)} 个地块)")

    def __len__(self):
        return len(self.valid_indices)
    

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        plot_id = row['plot_id']
        lai_label = row['LAI'] if 'LAI' in row else row['LAI']
        
        # --- 1. 点云数据处理 ---
        npy_path = os.path.join(self.data_root, f"{plot_id}.npy")
        points_all = np.load(npy_path) # 形状 (16384, 6)


         # --- 2. 影像数据处理 ---
        ms_path = os.path.join(self.ms_root, f"{plot_id}.npy")
        ms_img = np.load(ms_path) # shape: (5, 224, 224)

        # --- 3. 点云统计数据处理 ---
        stats_values = row[self.stats_cols].values.astype(np.float32)
        # Z-Score 标准化：(原始值 - 训练集均值) / 训练集标准差
        stats_norm = (stats_values - self.stats_mean) / self.stats_std

        # --- 4. 转换为 Tensor ---
        points_tensor = torch.from_numpy(points_all).float().permute(1, 0)
        stats_tensor = torch.from_numpy(stats_norm).float() # 维度: (35,)
        ms_tensor = torch.from_numpy(ms_img).float() # 维度: (5, 224, 224)
        label_tensor = torch.tensor(lai_label, dtype=torch.float32)

        return points_tensor, stats_tensor, ms_tensor, label_tensor