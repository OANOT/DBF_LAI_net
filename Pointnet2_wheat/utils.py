import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
def draw_scatter(predictions, target, x_label, y_label, min, max, save_path):
    plt.rcParams["font.family"] = ['Times New Roman', 'SimHei']
    plt.rcParams["font.size"] = 25  # 设置默认字体大小
    plt.rcParams["axes.titlesize"] = 22  # 设置标题字体大小
    plt.rcParams["axes.labelsize"] = 22  # 设置轴标签字体大小
    plt.rcParams["xtick.labelsize"] = 22  # 设置x轴刻度标签字体大小
    plt.rcParams["ytick.labelsize"] = 22  # 设置y轴刻度标签字体大小
    plt.rcParams['xtick.major.width'] = 2  # 设置X主刻度线的宽度
    plt.rcParams['ytick.major.width'] = 2  # 设置Y主刻度线的宽度
    plt.rcParams['xtick.major.size'] = 6  # x轴主刻度线长度
    plt.rcParams['ytick.major.size'] = 6  # y轴主刻度线长度
    plt.rcParams["legend.fontsize"] = 22  # 设置图例字体大小
    plt.figure(figsize=(8, 7))
    bwith = 2   # 设置边框大小
    ax = plt.gca()#获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    r2 = r2_score(target, predictions)
    rmse = np.sqrt(mean_squared_error(target, predictions))

    mae = np.mean(np.abs(target - predictions))
    if r2 < 0:
        r2 = 0

    coeff = np.polyfit(target, predictions, 1)  # 拟合线性关系
    y_fit = np.polyval(coeff, target)  # 计算拟合值
        # 绘制拟合曲线
    point_sizes = 150
    plt.scatter(target, predictions, color='#6D65A3',s = point_sizes,edgecolors='black')  # 绘制散点图
        # plt.plot(target, y_fit, color='red')  # 绘制拟合曲线
        # # 计算数据的标准差
    residuals = predictions - y_fit
    std_dev = np.std(residuals)

        # # 对target和y_fit进行排序
    # sort_index = np.argsort(target)
    # target_sorted = target[sort_index]
    # y_fit_sorted = y_fit[sort_index]

        # # 绘制±一倍标准差线
        # plt.fill_between(target_sorted, y_fit_sorted - std_dev, y_fit_sorted + std_dev, color='gray', alpha=0.3)
    plt.plot([min,max], [min,max], 'k--', lw=3)  # 绘制y=x曲线
    plt.xlabel(x_label)
    plt.ylabel(y_label)

        # 设置横坐标和纵坐标的范围
    plt.xlim(min,max)
        # plt.xticks([40, 50, 60, 70, 80,90])
    plt.ylim(min, max)    
        # plt.yticks([10, 20, 30, 40, 50])
        # 设置横纵坐标等间隔等距离绘制
    plt.gca().set_aspect('equal', adjustable='box')

        # # 在图上标出评价指标
    plt.text(0.50, 0.22, f'R² = {r2:.3f}', transform=plt.gca().transAxes)
    plt.text(0.50, 0.15, f'RMSE = {rmse:.3f}', transform=plt.gca().transAxes)
    plt.text(0.50, 0.08, f'MAE = {mae:.3f}', transform=plt.gca().transAxes)

        # # 保存图形时设置DPI
    plt.savefig(save_path, dpi=400)
    plt.show()
def calculate_metrics(predictions, target):
    r2 = r2_score(target, predictions)
    rmse = np.sqrt(mean_squared_error(target, predictions))
    
    # 计算相对 RMSE，单位为百分比
    mean_target = np.mean(target)
    rmse_percent = (rmse / mean_target) * 100
    
    return r2, rmse_percent
# 评估模型性能
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mean_y = np.mean(y_true)
    rrmse = (rmse / mean_y) * 100  # 计算相对 RMSE

    print(f"{model_name} 模型评估：")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"相对均方根误差 (rRMSE): {rrmse:.2f}%")
    print(f"决定系数 (R²): {r2:.3f}")
    print("-" * 30)