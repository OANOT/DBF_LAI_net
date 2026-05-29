import os
import laspy
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 全局参数配置，方便子进程调用
VOXEL_SIZE = 0.02
NUM_POINT = 16384 # 4096 16384

def process_single_file(args):
    """
    处理单个 LAS 文件的函数，供多进程调用
    args: (row, las_root, out_root)
    """
    row, las_root, out_root = args
    plot_id = row['plot_id']
    
    # 构建 LAS 文件路径
    date_part, id_part = plot_id.split('_')
    las_path = os.path.join(las_root, date_part, f"las{plot_id}.las")
    
    if not os.path.exists(las_path):
        return f"跳过 (文件不存在): {plot_id}"
        
    try:
        # 1. 读取 LAS
        las = laspy.read(las_path)
        x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)
        r = np.array(las.red) / 65535.0
        g = np.array(las.green) / 65535.0
        b = np.array(las.blue) / 65535.0
        c = c = np.array(las.classification).astype(np.float32) / 3.0
        points_all = np.vstack((x, y, z, r, g, b, c)).transpose()

        # 转为 open3d 格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_all[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points_all[:, 3:6])

        # 3. 估计法向量
        # search_param 定义了如何寻找邻域点：半径为 0.1，且最多找 30 个邻居
        # 你需要根据你的点云实际尺度（比如样方是米还是厘米）调整 radius 的大小
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=100))
  
        # 4. 体素降采样
        downpcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE) 
        down_pts = np.asarray(downpcd.points)
        down_colors = np.asarray(downpcd.colors)
        down_normals = np.asarray(downpcd.normals) 
        # voxel_points = np.hstack((down_pts, down_colors, down_normals))

        from scipy.spatial import cKDTree
        tree = cKDTree(points_all[:, :3])
        _, idx = tree.query(down_pts, k=1)
        down_classification = points_all[idx, 6] # 获取降采样后点对应的分类
        
        # 合并降采样后的所有特征：xyz, rgb, normals, classification
        voxel_points = np.hstack((down_pts, down_colors, down_normals, down_classification.reshape(-1, 1)))

        # 5. 对齐到固定的 num_point (4096)
        N_voxel = voxel_points.shape[0]
        if N_voxel >= NUM_POINT:
            pcd_fps = downpcd.farthest_point_down_sample(num_samples=NUM_POINT)
            selected_points = np.asarray(pcd_fps.points)
            selected_colors = np.asarray(pcd_fps.colors)
            selected_normals = np.asarray(pcd_fps.normals)   
            _, idx_fps = tree.query(selected_points, k=1)
            selected_classification = points_all[idx_fps, 6]
            
            # 最终合并：xyz, rgb, normals, classification
            final_points = np.hstack((selected_points, selected_colors, selected_normals, selected_classification.reshape(-1, 1)))
            # final_points = np.hstack((selected_points, selected_colors, selected_normals))
        else:
            # 点不足则随机重复补齐
            np.random.seed(0)
            choice = np.random.choice(N_voxel, NUM_POINT, replace=True)
            np.random.seed()
            final_points = voxel_points[choice, :]
        
        final_points[:, 0] -= np.mean(final_points[:, 0])
        final_points[:, 1] -= np.mean(final_points[:, 1])
        # Z 对齐地面 (底部归零)
        z_min = np.min(final_points[:, 2])
        final_points[:, 2] -= z_min 
        # # 统一物理比例缩放 (假设小麦最高 2m)
        # max_physical_height = 2.0 
        # final_points[:, :3] /= max_physical_height

        # # 6. 保存为 npy 文件
        out_name = f"{plot_id}.npy"
        out_path = os.path.join(out_root, out_name)
        np.save(out_path, final_points)

        # out_txt_name = f"{plot_id}.txt"
        # out_txt_path = os.path.join(out_root, 'txt', out_txt_name)
        # # fmt='%.15g' 保证坐标的高精度输出，避免默认6位小数导致的精度丢失
        # np.savetxt(out_txt_path, final_points, fmt='%.15g', delimiter=' ')
        
        return f"成功处理: {plot_id}"
        
    except Exception as e:
        return f"处理失败 {plot_id}: {str(e)}"


def process_and_save_parallel(csv_path, las_root, out_root):
    """
    并行处理主函数
    """
    os.makedirs(out_root, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # 准备传递给子进程的参数列表
    tasks = [(row, las_root, out_root) for _, row in df.iterrows()]
    
    # 获取 CPU 核心数，留 1-2 个核心给系统，避免电脑卡死
    max_workers = max(1, os.cpu_count() - 1)
    print(f"启动并行处理，使用 {max_workers} 个进程...")

    success_count = 0
    # 使用进程池并行执行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_file, task) for task in tasks]
        
        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            result_msg = future.result()
            print(result_msg) # 可选：打印每个文件的处理结果
            if "成功处理" in result_msg:
                success_count += 1
                
    print(f"\n全部任务完成！成功处理 {success_count}/{len(df)} 个文件。")


if __name__ == '__main__':
    # 注意：在 Windows 系统中，多进程代码必须放在 if __name__ == '__main__': 下运行
    las_path = 'H:/linux_disk/1_WCC/Pointcept/data/wheatPhenomics/pointcloud/02LasClip'
    process_and_save_parallel('LAI_data.csv', las_path, 'data/npy_data_LAI')