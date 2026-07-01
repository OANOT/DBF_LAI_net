import os
import sys
from wheat_util import DATA_PATH, collect_point_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/wheat_anno_paths.txt'))]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

output_folder = os.path.join(ROOT_DIR, 'data/wheat/wheat_seg_data')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    sample_las_path = sorted(
        [f for f in os.listdir(anno_path) if
         f.endswith('.las')],
        key=lambda x: int(x.split('las')[1].split('.')[0]))
    # 提取LAS文件的数字作为plot_id，此处构建了plot_id数组
    # plot_ids = [int(f.split('las')[1].split('.')[0]) for f in sample_las_path if f.endswith('.las')]
    plot_ids = [(f.split('las')[1].split('.')[0] ) for f in sample_las_path if f.endswith('.las')]

    for plot_id in plot_ids:
        try:
            print(plot_id)
            elements = anno_path.split('\\')
            # out_filename = elements[-1]+'_' + str(plot_id) + '.npy'
            out_filename = str(plot_id) + '.npy'  # Area_1_hallway_1.npy
            collect_point_label(os.path.join(anno_path, 'las'+str(plot_id)+'.las'), os.path.join(output_folder, out_filename), 'numpy')
        except:
            print(anno_path, 'ERROR!!')
