import os
import sys
import random
import glob
import yaml

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from util import minkowski_collate_fn

def get_generator(data_path, batch_size, workers, dataset):

    train_set = LiDAR_feature_dataset(data_path, set='train', dataset='nuscenes')
    val_set = LiDAR_feature_dataset(data_path, set='val', dataset='nuscenes')

    train_dataloader = DataLoader(
            train_set,
            batch_size=batch_size,
            collate_fn=minkowski_collate_fn,
            num_workers=workers,
            shuffle = True,
            drop_last = False
        )

    val_dataloader = DataLoader(
            val_set,
            batch_size=1,
            collate_fn=minkowski_collate_fn,
            num_workers=workers,
            shuffle = False,
            drop_last = False
        )
    
    return train_dataloader, val_dataloader

class LiDAR_feature_dataset(Dataset):
    def __init__(self, data_path, set='train', dataset='nuscenes'):

        assert dataset in ['semantickitti', 'nuscenes', 'waymo']
        assert set in ['train', 'valid', 'test'] 
    
        self.set = set
        self.dataset = dataset
        self.data_path = data_path

        self.im_idx=[]
        self.feat_timestamp=[]

        f = open(f'./dataset/split/{self.dataset}_{self.set}.txt', 'r')
        while True:
            line = f.readline()
            if not line: break
            self.feat_timestamp.append(line[:-1])
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.feat_timestamp)

    def __getitem__(self, ind):

        file_name = self.feat_timestamp[ind]

        if self.dataset == 'nuscenes':
            data = torch.load(os.path.join(self.data_path, f'{file_name}.pt'))
            multiview_feature = torch.load(os.path.join(self.feature_path, f'{file_name}.pt'))
            raw_data = data[0].T
            annotated_data = data[2]

        elif self.dataset == 'waymo':
            timestamp = file_name.split('_')[-1]
            scene_name = file_name[:-17]
            data = torch.load(os.path.join(self.data_path, f'{scene_name}/lidar/{timestamp}.pth'))
            multiview_feature = torch.load(os.path.join(self.feature_path, f'{file_name}.pt'))

            raw_data = data[0]
            annotated_data = data[2][:,1].int()
        
        remapped_labels = self.CLASS_REMAP[annotated_data]

        remapped_labels -= 1

        pcd_mask = multiview_feature['mask_full']

        feature = multiview_feature['feat']


        if self.dataset =='nuscenes':
            valid = remapped_labels != 255
            raw_data= raw_data[valid,:]
            remapped_labels = remapped_labels[valid]

        point = np.array(raw_data[:,:3])

        if self.set == 'train':
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                    np.cos(theta), 0], [0, 0, 1]])

            point = random.choice([np.array(np.dot(point, rot_mat) * scale_factor, dtype=np.float32), point])

        point_feature = torch.ones(point.shape[0], 3)

        return {"coords":torch.from_numpy(point), "feats": point_feature, 'feats_3d':feature, 'mask':pcd_mask, 'labels':torch.from_numpy(remapped_labels), 'dataset':self.dataset}


if __name__ == "__main__":

    import open3d as o3d

    def pointcloud_projection(lidars, image, proj, tag):
    
        color_temp = np.zeros_like(lidars[:,:3])

        img_array = np.array(image)


        for i, coord in enumerate(proj):
            if coord[0] >= img_array.shape[0] or coord[1] >= img_array.shape[1]:
                continue
            color_temp[i,:] = img_array[int(coord[0]), int(coord[1])]/255

        cloud = o3d.geometry.PointCloud()

        v3d = o3d.utility.Vector3dVector
        cloud.points = v3d(lidars[:,:3])
        cloud.colors = v3d(color_temp)

        o3d.io.write_point_cloud(f'pcds/projection_2_{tag}.pcd', cloud)

    #SemanticKITTI = LiDAR_feature_dataset(None, data_path='/root/code/nuscenes_single', feature_path='/root/code/nuscenes_openseg_features_1', set='train', dataset='nuscenes')
    SemanticKITTI = LiDAR_feature_dataset(None, data_path='/root/code/waymo', feature_path='/dataset/waymo_output', set='train', dataset='waymo')
    #SemanticKITTI = LiDAR_feature_dataset(None, data_path='/root/code/waymo', feature_path='/dataset/waymo_output', set='valid', dataset='waymo')

    img_idx = SemanticKITTI[100]





   
    
