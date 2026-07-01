import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
import torchvision.models as models

class get_model(nn.Module):
    def __init__(self, num_class=1, normal_channel=True, stats_dim=34, fusion_dim=256):
        super(get_model, self).__init__()
        
        self.normal_channel = normal_channel
        in_channel = 7 if normal_channel else 0 
        
        # # ================= step1. PointNet++ 几何流 (提取 1024维) =================
        self.sa1 = PointNetSetAbstractionMsg(256, [0.05, 0.1, 0.2], [8, 16, 64], in_channel,
                                             [[16, 16, 32], [32, 32, 64], [32, 48, 64]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [16, 64, 128], 160,
                                             [[32, 32, 64], [64, 64, 128], [64, 64, 128]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 320 + 3, [256, 512, 1024], True)

        self.pn_proj= nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),          
            nn.Linear(512, fusion_dim),
            nn.BatchNorm1d(fusion_dim), 
            nn.ReLU(),   
        )

        # ================= step2. 多光谱图像流  =================
        # 光谱特征投影层
        self.ms_spe_proj = nn.Sequential(
            # 第一层：4 -> 1024
            nn.Linear(4, 1024),  
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, fusion_dim), 
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )

        # ================================== step3. 模块注意力机制  ==================================
        # 观察2路原始特征，生成 2 个权重因子
        self.attention_net = nn.Sequential(
            nn.Linear(fusion_dim * 2,512),
            nn.BatchNorm1d(512),
            nn.ReLU(), 
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(), 
            nn.Linear(256, 2),
            nn.Softmax(dim=1) 
        )
        # ================================== step4. 最终回归头 ==================================
        # 💡 输入维度 = fusion_dim * 2 (即 256 * 2 = 512)
        self.final_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2,512),
            nn.BatchNorm1d(512),
            nn.ReLU(), 
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(), 
            nn.Linear(256, 1),
        )  

    def forward(self, xyz, stats, ms_imgs):
        # ------ 1. 点云特征 --------------------------
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)      
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) 
        _, l3_points = self.sa3(l2_xyz, l2_points) 
        x_pn2 = l3_points.view(B, 1024)
        x_fused_pc = self.pn_proj(x_pn2)

        # ------ 2. 光谱特征 --------------------------

        b, c, h, w = ms_imgs.size()
        # 将空间维度展平，4个波段作为序列长度
        spectral_input = ms_imgs.view(b, c, h * w)  # shape: [Batch, 4, H*W]
        data_with_nan = spectral_input.masked_fill(spectral_input == 0, float('nan'))
        pooled_output = torch.nanmean(data_with_nan, dim=2, keepdim=False)
        pooled_output = torch.nan_to_num(pooled_output, nan=0.0) # shape: [Batch, 4]
        x_fused_ms = self.ms_spe_proj(pooled_output)  # shape: [Batch, 256]

        # ------ 3. 特征融合 (拼接) --------------------------
        x_all = torch.cat([x_fused_pc, x_fused_ms], dim=1) # shape: [Batch, fusion_dim * 2]
        weights = self.attention_net(x_all) 
        w_pc = weights[:, 0:1]
        w_ms = weights[:, 1:2]
        x_fused = torch.cat([w_pc * x_fused_pc,w_ms * x_fused_ms], dim=1) 

        # ------ 4. 最终回归 --------------------------
        x = self.final_mlp(x_fused)
         
        return x, _
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.criterion = nn.SmoothL1Loss() 

    def forward(self, pred, target):
        target = target.view(-1, 1).float()
        total_loss = self.criterion(pred, target)
        return total_loss