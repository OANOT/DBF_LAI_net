import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class WheatDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        # plots = sorted(os.listdir(data_root))
        # train_data, test_data = train_test_split(plots, test_size=0.3, random_state=42)
        # if split == 'train':
        #     plot_split = train_data
        # else:
        #     plot_split = test_data
        plot_split = sorted(os.listdir(data_root))
        self.plot_points, self.plot_labels = [], []
        self.plot_coord_min, self.plot_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(3)

        for plot in tqdm(plot_split, total=len(plot_split)):
            plot_path = os.path.join(data_root, plot)
            plot_data = np.load(plot_path)  # xyzrgbl, N*7
            points, labels = plot_data[:, 0:6], plot_data[:, 6]  # xyzrgb, N*6; l, N
            # labels = np.where(labels == 2, 0, labels)
            labels = labels - 1
            tmp, _ = np.histogram(labels, range(4))  # 需修改！！！
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.plot_points.append(points), self.plot_labels.append(labels)
            self.plot_coord_min.append(coord_min), self.plot_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 2.0)
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        plot_idxs = []
        for index in range(len(plot_split)):
            plot_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.plot_idxs = np.array(plot_idxs)
        print("Totally {} samples in {} set.".format(len(self.plot_idxs), split))

    def __getitem__(self, idx):
        plot_idx = self.plot_idxs[idx]
        points = self.plot_points[plot_idx]   # N * 6
        labels = self.plot_labels[plot_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.plot_coord_max[plot_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.plot_coord_max[plot_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.plot_coord_max[plot_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        # selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.plot_idxs)

class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        np.random.seed(42)
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        plots = sorted(os.listdir(root))
        train_data, test_data = train_test_split(plots, test_size=0.3, random_state=42)
        if split == 'train':
            self.file_list = train_data
        else:
            self.file_list = test_data
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.plot_coord_min, self.plot_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            labels = data[:, 6]  # xyzrgb, N*6; l, N
            labels = np.where(labels == 2, 0, labels)
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(labels)
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.plot_coord_min.append(coord_min), self.plot_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(2)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(3))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 2.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_plot, label_plot, sample_weight, index_plot = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_plot = np.vstack([data_plot, data_batch]) if data_plot.size else data_batch
                label_plot = np.hstack([label_plot, label_batch]) if label_plot.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_plot.size else batch_weight
                index_plot = np.hstack([index_plot, point_idxs]) if index_plot.size else point_idxs
        data_plot = data_plot.reshape((-1, self.block_points, data_plot.shape[1]))
        label_plot = label_plot.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_plot = index_plot.reshape((-1, self.block_points))
        return data_plot, label_plot, sample_weight, index_plot

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = WheatDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in [1, 2]:
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()