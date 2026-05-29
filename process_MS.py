import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import cv2
from tqdm import tqdm

def preprocess_ms_multiyr(tif_paths, shp_mapping, output_dir, canvas_size=512):
    """
    适配多年份的预处理逻辑
    参数:
    - tif_paths: TIFF 文件路径列表
    - shp_mapping: 字典类型，例如 {"2023": "path_to_2023.shp", "2024": "path_to_2024.shp"}
    - output_dir: 输出目录
    - canvas_size: 固定画布大小
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 用于缓存已经读取的 GeoDataFrame，避免重复读取同一 SHP
    gdf_cache = {}

    for tif_path in tif_paths:
        filename = os.path.basename(tif_path)
        date_prefix = filename.split('_')[0] # 提取如 "20230425"
        year = date_prefix[:4]               # 提取如 "2023"
        
        # 1. 根据年份获取对应的 SHP 路径
        if year not in shp_mapping:
            print(f"⚠️ 跳过 {filename}: 未找到 {year} 年对应的 SHP 映射。")
            continue
        
        shp_path = shp_mapping[year]

        # 2. 读取或从缓存获取 GDF
        if year not in gdf_cache:
            print(f"📖 正在加载 {year} 年矢量文件: {shp_path}")
            gdf_cache[year] = gpd.read_file(shp_path, engine="pyogrio")
        
        gdf = gdf_cache[year]

        # 3. 处理影像
        with rasterio.open(tif_path) as src:
            # 统一坐标系 (CRS)
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            
            for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc=f"Processing {date_prefix}"):
                # 💡 注意：请确认 2023 和 2024 的 SHP 属性表中地块 ID 列名是否都叫 'Id'
                plot_id = row['Id'] 
                geom = [row.geometry]
                try:
                    # 掩膜裁切
                    out_image, _ = mask(src, geom, crop=True)
                    
                    # 归一化 (根据 16-bit 还是 8-bit 修改除数)
                    ms_data = out_image.astype(np.float32) 
                    ms_data = np.clip(ms_data, 0, 1)


                    if ms_data.shape[0] == 5:
                        ms_data = ms_data[1:, :, :] # 假设切掉的是 Blue，保留 G, R, RE, NIR
                    # 转换维度 (B, H, W) -> (H, W, B)

                    ms_data = ms_data.transpose(1, 2, 0)
                    h, w, c = ms_data.shape

                    # 4. 固定画布填充 (Padding)
                    canvas = np.zeros((canvas_size, canvas_size, c), dtype=np.float32)
                    

                    # 判断是否需要裁剪 (数据比画布大) 还是填充 (数据比画布小)
                    if h > canvas_size and w > canvas_size:
                        # --- 情况 A: 裁剪 ---
                        # 计算裁剪的起始点 (中心对齐)
                        y_start = (h - canvas_size) // 2
                        x_start = (w - canvas_size) // 2
                        
                        # 计算裁剪的结束点
                        y_end = y_start + canvas_size
                        x_end = x_start + canvas_size
                        
                        # 从 ms_data 中心裁剪并赋值给 canvas
                        canvas = ms_data[y_start:y_end, x_start:x_end, :]
                    elif h > canvas_size and w <= canvas_size:
                        # --- 情况 A: 裁剪 ---
                        # 计算裁剪的起始点 (中心对齐)
                        y_start = (h - canvas_size) // 2
                        x_start = (canvas_size - w) // 2
                        
                        # 计算裁剪的结束点
                        y_end = y_start + canvas_size
                        x_end = x_start + w
                        
                        # 从 ms_data 中心裁剪并赋值给 canvas
                        canvas[:, x_start:x_end, :] = ms_data[y_start:y_end, :, :]
                    elif h <= canvas_size and w > canvas_size:
                        # --- 情况 A: 裁剪 ---
                        # 计算裁剪的起始点 (中心对齐)
                        y_start = (canvas_size - h) // 2
                        x_start = (w - canvas_size) // 2
                        
                        # 计算裁剪的结束点
                        y_end = y_start + h
                        x_end = x_start + canvas_size
                        
                        # 从 ms_data 中心裁剪并赋值给 canvas
                        canvas[y_start:y_end, :, :] = ms_data[:, x_start:x_end, :]

                    else:
                        # --- 情况 B: 填充 ---
                        # 计算粘贴的起始点 (居中)
                        y_start = (canvas_size - h) // 2
                        x_start = (canvas_size - w) // 2
                        
                        # 计算粘贴的结束点
                        y_end = y_start + h
                        x_end = x_start + w
                        
                        # 将 ms_data 放置在 canvas 中心
                        canvas[y_start:y_end, x_start:x_end, :] = ms_data


 

                    # 5. 转换回 PyTorch 格式 (B, H, W) 并保存
                    final_data = canvas.transpose(2, 0, 1)
                    save_name = f"{date_prefix}_{plot_id}.npy"
                    np.save(os.path.join(output_dir, save_name), final_data)
                    
                except ValueError:
                    # 几何体不在当前影像范围内
                    print(f"⚠️ 地块 {plot_id} 不在影像范围内，已跳过。")
                    continue

    print(f"\n✅ 全部处理完成！存至: {output_dir}")

# --- 配置区 ---
# 1. 定义年份与 SHP 的对应关系
year_shp_map = {
    "2023": "H:/linux_disk/1_WCC/Pointcept/data/wheatPhenomics/vector/2023_sample_plot_820.shp",
    "2024": "H:/linux_disk/1_WCC/Pointcept/data/wheatPhenomics/vector/2024_sample_plot_677.shp"
}

# 2. 列出所有需要处理的影像
tif_files = [
    'G:/SDAU_DGP/20230324/20230324_MS_mask.tif',
    'G:/SDAU_DGP/20230426/20230426_MS_mask.tif',
    'G:/SDAU_DGP/20240404/20240404_MS_mask.tif',
    'G:/SDAU_DGP/20240409/20240409_MS_mask.tif',
    'G:/SDAU_DGP/20240414/20240414_MS_mask.tif',
    'G:/SDAU_DGP/20240419/20240419_MS_mask.tif',
    'G:/SDAU_DGP/20240424/20240424_MS_mask.tif',

]

# 3. 运行
preprocess_ms_multiyr(
    tif_paths = tif_files,
    shp_mapping = year_shp_map,
    output_dir = 'data/ms_npy_LAI',
    canvas_size = 512 # 设为 600 像素以兼容 514 宽度的样方并预留余量
)