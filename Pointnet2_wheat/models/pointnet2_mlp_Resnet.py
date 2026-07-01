import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
import torchvision.models as models

class get_model(nn.Module):
    def __init__(self, num_class=1, normal_channel=True, stats_dim=34, fusion_dim=256):
        super(get_model, self).__init__()
        
        self.normal_channel = normal_channel
        # in_channel是除了xyz外的特征维度
        in_channel = 6 if normal_channel else 0 
        
        # --- 支流 1：PointNet++ 几何流 (提取 1024维) ---
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        
        # 💡 对齐层 1：1024 -> 256
        self.pc_proj = nn.Sequential(nn.Linear(1024, fusion_dim), nn.BatchNorm1d(fusion_dim), nn.ReLU())

        # --- 支流 2：多光谱图像流 (ResNet18 提取 512维) ---
        ## ================= 分支一：空间特征提取 (ResNet18) =================
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Identity() 
        self.ms_branch = resnet
        
        # 💡 对齐层 3：512 -> 256
        self.ms_proj = nn.Sequential(nn.Linear(512, fusion_dim), nn.BatchNorm1d(fusion_dim), nn.ReLU())
        ## ================= 分支二：光谱特征提取 (1D-CNN) =================
        # 将5个波段视为长度为5的序列进行一维卷积
        self.spectral_conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 全局池化层，用于压缩 1D-CNN 展平后的空间维度
        # self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 光谱特征投影层
        self.spectral_proj = nn.Sequential(nn.Linear(64, fusion_dim), nn.BatchNorm1d(fusion_dim), nn.ReLU())

        # 融合后的总维度是两个分支维度之和
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )


        # --- 💡 模块注意力机制 ---
        # 观察2路原始特征，生成 2 个权重因子
        self.attention_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1) 
        )
        
        # --- 最终回归头 ---
        # 💡 输入维度 = fusion_dim * 2 (即 256 * 2 = 512)
        self.fc1 = nn.Linear(fusion_dim * 2, 512) 
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, xyz, stats, ms_imgs):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
            
        # 1. 各分支特征提取与对齐 (全部对齐到 fusion_dim)
        _, l3_points = self.sa1(xyz, norm)
        _, l3_points = self.sa2(_, l3_points)
        _, l3_points = self.sa3(_, l3_points)
        x_pc = self.pc_proj(l3_points.view(B, 1024))
        
        # x_stats = self.stats_mlp(stats)
        
        x_ms_spa = self.ms_proj(self.ms_branch(ms_imgs)) 

        b, c, h, w = ms_imgs.size()
        # 将空间维度展平，5个波段作为序列长度
        spectral_input = ms_imgs.view(b, c, h * w) # shape: [Batch, 5, H*W]
        
        spectral_feat = self.spectral_conv(spectral_input) # shape: [Batch, 64, H*W]

        # # 增加维度适配池化层，并压缩空间特征
        # spectral_feat = spectral_feat.unsqueeze(-1) 
        # spectral_feat = self.global_pool(spectral_feat) # shape: [Batch, 64, 1, 1]
        # spectral_feat = spectral_feat.view(b, -1) # shape: [Batch, 64]
        # x_ms_spe = self.spectral_proj(spectral_feat) # shape: [Batch, fusion_dim]

        data_with_nan = spectral_input.masked_fill(spectral_input == 0, float('nan'))
        pooled_output = torch.nanmean(data_with_nan, dim=2, keepdim=True)
        pooled_output = torch.nan_to_num(pooled_output, nan=0.0) # shape: [Batch, 4, 1]
        spectral_feat = self.spectral_conv(pooled_output) # shape: [Batch, 64, 1]
        spectral_feat = spectral_feat.view(b, -1) # shape: [Batch, 64]
        x_ms_spe = self.spectral_proj(spectral_feat) # shape: [Batch, fusion_dim]

        # 3. 特征融合 (拼接)
        x_ms = torch.cat([x_ms_spa, x_ms_spe], dim=1) # shape: [Batch, fusion_dim * 2]
        x_ms =self.final_fusion(x_ms)
        # 2. 计算模块注意力权重
        # 拼接用于观察的原始对齐特征
        combined_obs = torch.cat([x_pc, x_ms], dim=1) # B x 768
        weights = self.attention_net(combined_obs) # B x 3

        w_pc = weights[:, 0:1]
        w_ms = weights[:, 1:2]

        # w_pc = weights[:, 0:1]
        # w_stats = weights[:, 1:2]
        # w_ms_spa = weights[:, 2:3]
        # w_ms_spe = weights[:, 3:4]
        
        # 3. 💡 核心修改：加权后进行拼接
        # 此时每一个 256 维的特征块都被其对应的权重缩放
        x_fused = torch.cat([w_pc * x_pc, w_ms * x_ms], dim=1) # B x 768
        # x_fused = torch.cat([w_pc * x_pc, w_stats * x_stats, w_ms_spa * x_ms_spa, w_ms_spe * x_ms_spe], dim=1) # B x 768
        
        # 4. 最终回归
        x = self.drop1(F.relu(self.bn1(self.fc1(x_fused))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x, l3_points

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.criterion = nn.SmoothL1Loss() 

    def forward(self, pred, target):
        target = target.view(-1, 1).float()
        total_loss = self.criterion(pred, target)
        return total_loss