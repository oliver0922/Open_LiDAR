import os
import torch
import numpy as np
#import open_clip
import clip
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('.')
#from sort.sort import *

import tensorflow as tf2
import tensorflow.compat.v1 as tf
from util.fusion_util import extract_openseg_img_feature
from util.mapper import PointCloudToImageMapper

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def text_extractor(text_set, device, normalize_feature=True): 
    model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
    model = model.to(device)
    text_set = [f"a {c} in a scene" for c in text_set]
    text_set = clip.tokenize(text_set).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_set)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

class OpensegExtracter():
    def __init__(self, dim=768, device="cuda:0"):

        self.dim = dim
        self.openseg_model = tf2.saved_model.load('/root/code/open_lidar/openseg', tags=[tf.saved_model.tag_constants.SERVING],)
        self.text_emb = tf.zeros([1, 1, dim])

        self.mapper = PointCloudToImageMapper(image_dim=(800, 450), cut_bound=5)
        self.device = device
        self.dim = 768


    def extract(self, metadata):
        points, img_paths, intrs, poses, label = metadata['coords'], metadata['colors_path'], metadata['intrinsics'], metadata['poses'], metadata['labels']
        lidar_path = metadata['coords_path']
        assert len(intrs) == len(poses)

        locs_in = points[0].float()
        labels_in = label[0].squeeze()
        mask_entire = (labels_in!=255).to(self.device)

        locs_in = locs_in[mask_entire]
        n_points = locs_in.shape[0]

        n_points_cur = n_points
        counter = torch.zeros((n_points_cur, 1), device=self.device)
        sum_features = torch.zeros((n_points_cur, self.dim), device=self.device)

        num_img = len(intrs)

        vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=self.device)


        os.makedirs( "/".join(img_paths[0][0].split('/')[:-1]).replace('color', 'openseg'), exist_ok=True)
        img_id = 0
        for intr, img_dir, pose in zip(intrs, img_paths, poses):

            mapping = np.ones([n_points_cur, 4], dtype=int)
            mapping[:, 1:4] = self.mapper.compute_mapping(pose[0], locs_in, depth=None, intrinsic=intr[0])
            if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
                continue

            mapping = torch.from_numpy(mapping).to(self.device)
            mask = mapping[:, 3]
            vis_id[:, img_id] = mask

            feat_2d = extract_openseg_img_feature(img_dir[0], self.openseg_model, self.text_emb, img_size=[450, 800])

            feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0).to(self.device)

            counter[mask!=0]+= 1
            sum_features[mask!=0] += feat_2d_3d[mask!=0]
            
            img_id += 1 

        counter[counter==0] = 1e-5
        feat_bank = sum_features/counter
        point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

        mask = torch.zeros(n_points, dtype=torch.bool).to(self.device)
        mask[point_ids] = True
        mask_entire[mask_entire==True] = mask

        np.savez_compressed(lidar_path[0].replace('/lidar/','/openseg/').replace('bin','npz'), feat = feat_bank[mask].half().cpu().numpy(), mask_full = mask_entire.cpu().numpy())

if __name__ == "__main__":
    from data.dataloader import get_generator, LiDARDataset
    import sys
    from tqdm import tqdm
    

    extracter = OpensegExtracter(device="cuda")

    dataset = LiDARDataset('waymo', '/dataset/Waymo_split', split='val')
    if int(sys.argv[1]) in [0,1,2]:
        dataset = LiDARDataset('waymo', '/dataset/Waymo_split', split='train', split_int=int(sys.argv[1]))
    else:        
        dataset = LiDARDataset('waymo', '/dataset/Waymo_split', split='val', split_int=int(sys.argv[1]))

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=24,
        shuffle = False,
        drop_last = False
    )
    pbar = tqdm(dataloader, desc = f"Openseg Extraction Device")
    for idx, data in enumerate(pbar):
        extracter.extract(data)
    