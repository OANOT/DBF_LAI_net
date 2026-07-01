import os
import torch
import numpy as np
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from data_utils.FAPARDataLoader import WheatFAPARDataset
from utils import draw_scatter
import sys
import pandas as pd
from datetime import datetime
import random

def seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 强制锁定算法
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 这一行会配合上面的环境变量，彻底消除 CuBLAS 的不确定性
    # 加上 warn_only=True 是为了防止某些算子由于没有确定性实现而直接报错崩溃
    torch.use_deterministic_algorithms(True, warn_only=True) 
seed_everything(42)  # 设置随机种子，确保结果可复现
def evaluate_and_plot(model_path, data_root, csv_path, model_name='pointnet2_mlp'):
    """
    主评估函数
    """
    # --- 环境配置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_point = 16384 # 需与训练时一致
    num_classes = 1

    # --- 加载测试数据 ---
    print("🚀 正在加载测试集...")
    TEST_DATASET = WheatFAPARDataset(
        csv_path=csv_path, 
        data_root=data_root, 
        split='test',  # 确保是 8:1:1 中的 Test 部分
        num_point=num_point
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=16, shuffle=False, num_workers=4
    )

    # --- 加载模型权重 ---
    print(f"🛠️ 正在初始化模型: {model_name}...")
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_classes).to(device)
    
    print(f"📂 正在加载权重文件: {model_path}...")
    checkpoint = torch.load(model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    # --- 执行推理 ---
    y_test_list = []
    y_pred_list = []

    all_test_points = []
    all_test_targets = []
    
    print("🏃 正在进行测试集预测并导出原始数据...")
    with torch.no_grad():
        classifier.eval()
        test_pred, test_gt = [], []
        # 💡 同样解包三个值
        for points, stats, target in tqdm(testDataLoader, total=len(testDataLoader), desc='Final Testing'):
            points, stats, target = points.float().cuda(), stats.float().cuda(), target.float().cuda()
            
            # 💡 传入两个参数
            pred, _ = classifier(points, stats)
            
            test_pred.extend(pred.cpu().numpy().flatten())
            test_gt.extend(target.cpu().numpy().flatten())

        test_pred = np.clip(np.array(test_pred), 0.0, 1.0)
        test_gt = np.array(test_gt)
        
        # 计算测试集最终指标
        test_rmse = np.sqrt(np.mean((test_pred - test_gt) ** 2))
        test_mae = np.mean(np.abs(test_pred - test_gt))
        ss_res = np.sum((test_gt - test_pred) ** 2)
        ss_tot = np.sum((test_gt - np.mean(test_gt)) ** 2)
        test_r2 = 1 - (ss_res / (ss_tot + 1e-8))
    print("\n" + "="*30)
    print(f"✅ 测试完成!")
    print(f"测试集 R²   : {test_r2:.4f}")
    print(f"测试集 RMSE : {test_rmse:.4f}")
    print("="*30)
    results_df = pd.DataFrame({
            'True_FAPAR': test_pred,
            'Predicted_FAPAR': test_r2
        })
    results_df.to_csv('test_results2.csv', index=False)
    # --- 调用绘图函数 ---
    draw_scatter(test_pred, test_gt, x_label='True FAPAR', y_label='Predicted FAPAR', min=0, max=1, save_path='check_scatter.png')

# --- 执行入口 ---
if __name__ == '__main__':
    # 请根据你的实际路径修改以下变量

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'models'))
    MODEL_PATH = 'Pointnet2_wheat/log/sem_seg/2026-04-06_22-51/checkpoints/best_model.pth' # 改为你自己的路径
    DATA_ROOT = 'npy_data'
    CSV_FILE = 'FAPAR_data.csv'
    
    evaluate_and_plot(MODEL_PATH, DATA_ROOT, CSV_FILE)