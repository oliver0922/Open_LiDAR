from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import clip
import open_clip
import torch
import os
import sys
from glob import glob
import open3d as o3d
import tqdm
from matplotlib import image
import numpy as np
import sparse
import MinkowskiEngine as ME

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_o3d_pointcloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.array(color))
    return pcd

def scene_loader(lidar_path):
    scene = np.fromfile(lidar_path).reshape(-1,4)[:,:3]
    pcd = make_o3d_pointcloud(np.asarray(scene))
    return pcd


class LiDARDataset(Dataset):
    """LiDAR Dataset for nuScenes, Waymo

    Returns:
        coords, feats, labels, dataset
    """
    def __init__(self, dataset, dataset_dir, split='train', split_int = None, multisweep=False):
        #TODO: change directory structure with dataset splits
        super().__init__()
        
        assert dataset in ['waymo', 'nuscenes', 'semantickitti']
        assert split in ['train', 'val', 'test', 'all']

        self.dataset_dir = dataset_dir
        self.dataset = dataset
        self.scene_list = os.listdir(dataset_dir)
        self.multisweep = multisweep
        
        if multisweep:
            self.lidar_term = 'multi_lidar'
        else:
            self.lidar_term = 'lidar'
        self.lidar_list = sorted(glob(dataset_dir + f'/{split}/*/{self.lidar_term}/*.bin'))

        self.timestamp_list = [lidar_path.split('/')[-1].split('.')[0] for lidar_path in self.lidar_list]
        self.split = split
        if dataset == 'waymo':
            self.camera_names = ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_SIDE_LEFT', 'CAM_SIDE_RIGHT']
            self.map = {0: 255, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19, 20:20, 21:21, 22:22}

        elif dataset == 'nuscenes':
            self.camera_names = ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
            self.map = {1: 255, 5: 255, 7: 255, 8: 255, 10: 255, 11: 255, 13: 255, 19: 255, 20: 255, 0: 255,
  29: 255, 31: 255, 9: 0, 14: 1, 15: 2, 16: 2, 17: 3, 18: 4, 21: 5, 2: 6, 3: 6, 4: 6, 6: 6,
  12: 7, 22: 8, 23: 9, 24: 10, 25: 11, 26: 12, 27: 13, 28: 14, 30: 15}
            
        self.axes_transformation = np.array([[0,-1,0,0],
                                             [0,0,-1,0],
                                             [1,0,0,0],
                                             [0,0,0,1]])
        self.multisweep = multisweep
        

    def __len__(self) -> int:
        return len(self.lidar_list)
    
    def __getitem__(self, index) -> tuple:
        # LiDAR points
        lidar_path = self.lidar_list[index]

        if self.dataset == 'nuscenes' and not self.multisweep:
            point = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        elif self.dataset == 'waymo' and not self.multisweep:
            point = np.fromfile(lidar_path, dtype=np.float64).reshape(-1, 3)[:, :3]
        else:
            point = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 3)[:, :3]

        # Color images
        color_image_path_stem = lidar_path.split('.')[0].replace(f'/{self.lidar_term}', '/color')
        color_image_paths = [color_image_path_stem + f'_{camera_name}.jpg' for camera_name in self.camera_names]
        

        # CLIP features 
        clip_feature_path = lidar_path.split('.')[0].replace(f'/{self.lidar_term}', '/openseg')
        clip_feature = torch.tensor(np.load(clip_feature_path+'.npz')['feat'], dtype=torch.float16)

        # entire_mask features 
        entire_mask = torch.tensor(np.load(clip_feature_path+'.npz')['mask_full'])

        if self.multisweep:
            mask = torch.tensor(np.zeros_like(point[:,0]))
            mask[:len(entire_mask)] = entire_mask
            entire_mask = mask

        # Camera intrinsics
        K_path_stem = os.path.dirname(lidar_path).replace(f'/{self.lidar_term}', '/K')
        intrinsics= [np.load(K_path_stem + f'/{camera_name}.npy') for camera_name in self.camera_names]

        # LiDAR pose
        lidar_pose_stem = lidar_path.split('.')[0].replace(f'/{self.lidar_term}', '/lidar_pose')
        lidar_pose = np.loadtxt(lidar_pose_stem + '.txt')

        # labels
        if self.split != 'test':
            label_path = lidar_path.split('.')[0].replace(f'/{self.lidar_term}', '/label')
            label = np.fromfile(label_path + '.label', dtype=np.uint8).reshape(-1, 1)
            label = np.vectorize(self.map.__getitem__)(label)
        else : 
            label == None
        
        if self.multisweep:
            label_ = np.ones_like(point[:,0])*255
            label_[:len(label)] = label
            label = label_


        # Masks
        mask_path_stem = lidar_path.split('.')[0].replace(f'/{self.lidar_term}', '/mask')
        mask_paths = [mask_path_stem + f'_{camera_name}.npz' for camera_name in self.camera_names]
        masks = [sparse.load_npz(mask_path).todense() for mask_path in mask_paths]
        
        # Camera poses
        pose_stem = lidar_path.split('.')[0].replace(f'/{self.lidar_term}', '/pose')
        pose_paths = [pose_stem + f'_{camera_name}.txt' for camera_name in self.camera_names]
        poses = [np.loadtxt(pose_path) for pose_path in pose_paths]
        
        #point = point[label.squeeze() != -1,:]
        #label = label[label.squeeze() != -1]
        
        import pdb; pdb.set_trace()

        return {"coords":torch.from_numpy(point), 
                "colors_path": color_image_paths,
                "coords_path": lidar_path,
                "feats": torch.ones_like(torch.from_numpy(point)),
                "intrinsics" : intrinsics, 
                "clip": clip_feature,
                "mask": entire_mask,
                "poses" : poses,
                "lidar_pose" : lidar_pose, 
                'labels':torch.from_numpy(label),
                'dataset':self.dataset}
    
def minkowski_collate_fn(list_data):
    r"""
    Collation function for MinkowskiEngine.SparseTensor that creates batched
    cooordinates given a list of dictionaries.
    """
    coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(
        [d["coords"] for d in list_data],
        [d["feats"] for d in list_data],
        [d["labels"] for d in list_data],
        dtype=torch.float32,
    )
    clip_batch = torch.cat([d["clip"] for d in list_data], dim=0)

    import pdb; pdb.set_trace()

    _, mask_batch, _ = ME.utils.sparse_collate(
        [d["coords"] for d in list_data],
        [d["mask"] for d in list_data],
        [d["labels"] for d in list_data],
        dtype=torch.float32,
    )
    
    return {
        "coords": coords_batch,
        "feats" : feats_batch,
        "clip_feats" : clip_batch,
        "masks" : mask_batch,
        "labels": labels_batch
    }

def get_generator(data_path, batch_size, workers, dataset, multisweep=False):
    train_set = LiDARDataset(dataset, data_path, split='train', multisweep=multisweep)
    val_set = LiDARDataset(dataset, data_path, split='val', multisweep=multisweep)

    train_dataloader = DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            shuffle = True,
            drop_last = False
        )

    val_dataloader = DataLoader(
            val_set,
            batch_size=1,
            num_workers=workers,
            pin_memory=True,
            shuffle = False,
            drop_last = False
        )
    
    return train_dataloader, val_dataloader

def get_mink_generator(data_path, batch_size, workers, dataset, multisweep=False):

    train_set = LiDARDataset(dataset, data_path, split='train', multisweep=multisweep)
    val_set = LiDARDataset(dataset, data_path, split='val', multisweep=multisweep)

    train_dataloader = DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=workers,
            shuffle = False,
            collate_fn = minkowski_collate_fn,
            drop_last = False
        )

    val_dataloader = DataLoader(
            val_set,
            batch_size=1,
            num_workers=workers,
            shuffle = False,
            collate_fn = minkowski_collate_fn,
            drop_last = False
        )
    
    return train_dataloader, val_dataloader


if __name__ == '__main__':
    train_dataset = LiDARDataset(dataset='nuscenes', dataset_dir='/dataset/nuScene_lidarseg_split', split='train', multisweep=True)
    data = train_dataset[0]
