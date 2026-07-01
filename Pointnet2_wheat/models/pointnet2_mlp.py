import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self, num_class=1, normal_channel=True, stats_dim=34): # num_class对于回归固定为1
        super(get_model, self).__init__()
        
        # 如果输入总共6维(XYZ+3特征)，in_channel就是3
        # 如果输入总共9维(XYZ+6特征)，in_channel必须设为6
        self.normal_channel = normal_channel
        in_channel = 3 if normal_channel else 0 
        
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        self.stats_mlp = nn.Sequential(
            nn.Linear(stats_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(1024 + 128, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, xyz, stats):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
            
        # 提取点云特征
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x_pointcloud = l3_points.view(B, 1024)
        
        # 提取统计指标特征
        x_stats = self.stats_mlp(stats)
        
        # 特征融合 (Concatenation)
        x = torch.cat([x_pointcloud, x_stats], dim=1)
        
        # 最终回归
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x, l3_points

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        # --- 核心修改2：回归任务使用 MSE 或 SmoothL1 ---
        self.criterion = nn.SmoothL1Loss() 

    def forward(self, pred, target, trans_feat=None):
        # 确保 pred 和 target 都是 (Batch, 1)
        target = target.view(-1, 1).float()
        total_loss = self.criterion(pred, target)
        return total_loss