import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        self.k = 1  # 【修改点1】回归任务，输出维度固定为1 (FAPAR值)，不再使用 num_class
        # 注意：PointNetEncoder 的 channel=9 保持不变，因为你的 DataLoader 提供了9个通道
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=9)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)  # (B,1024)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        # ⭐ FAPAR范围约束（强烈建议）
        # x = torch.sigmoid(x)

        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        # --- 【修改点3】：使用均方误差损失 (MSE) ---
        self.criterion = torch.nn.SmoothL1Loss()

    def forward(self, pred, target, trans_feat):
        # --- 【修改点4】：删除 weight 参数，改用 MSE ---
        # pred 形状: (Batch, 1)
        # target 形状: (Batch) -> MSE 会自动处理广播机制，或者你可以 target.view(-1, 1)
        
        loss = self.criterion(pred, target.view(-1, 1))
        
        # 保留特征变换正则化，防止过拟合
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == '__main__':
    # 测试代码
    model = get_model(10) # num_class 参数现在实际上被忽略了
    xyz = torch.rand(12, 9, 2048) # 输入 (Batch, 9通道, 点数)
    pred, trans = model(xyz)
    print("输出形状:", pred.shape) # 应该输出 torch.Size([12, 1])