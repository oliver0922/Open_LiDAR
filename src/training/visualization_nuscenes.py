import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import open_clip
import open3d as o3d
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist
import imageio

# from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import argparse
from glob import glob
from tqdm import tqdm
import sparse

camera_names = ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']

device = "cuda" if torch.cuda.is_available() else "cpu"

class PointCloudToImageMapper(object):
    def __init__(self, image_dim,
            visibility_threshold=0.25, cut_bound=0, intrinsics=None):
        
        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        if self.intrinsics is not None: # global intrinsics
            intrinsic = self.intrinsics

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int) # simply round the projected coordinates
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                    * (pi[0] < self.image_dim[0]-self.cut_bound) \
                    * (pi[1] < self.image_dim[1]-self.cut_bound)
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                    - p[2][inside_mask]) <= \
                                    self.vis_thres * depth_cur

            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2]>0 # make sure the depth is in front
            inside_mask = front_mask*inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T

NUSCENES_PREDEFINED = (
    'barrier',
    'barricade',
    'bicycle',
    'bus',
    'car',
    'bulldozer',
    'excavator',
    'concrete mixer',
    'crane',
    'dump truck',
    'motorcycle',
    'pedestrian',
    'person',
    'traffic cone',
    'trailer',
    'semi trailer',
    'cargo container',
    'shipping container',
    'freight container',
    'truck',
    'road',
    'curb',
    'traffic island',
    'traffic median',
    'sidewalk',
    'grass',
    'grassland',
    'lawn',
    'meadow',
    'turf',
    'sod',
    'building',
    'wall',
    'pole',
    'awning',
    'tree',
    'trunk',
    'tree trunk',
    'bush',
    'shrub',
    'plant',
    'flower',
    'woods'
)

def make_o3d_pointcloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.array(color))
    return pcd

def text_extractor(normalize_feature=True): 
    # model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='./ovseg_clip_l_9a1909.pth')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c.split('.')[-1]}") for c in NUSCENES_FULL_CLASSES]).to(device)
    text_inputs = torch.cat([tokenizer(f"a photo of a {c}") for c in NUSCENES_PREDEFINED]).to(device)

    text_features = model.encode_text(text_inputs)
    if normalize_feature:
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

def image_loader(path):
    image_path = './color'
    image_feature, mask_full = torch.load(image_path).values()
    return image_feature, mask_full

def scene_loader(lidar_path):
    scene = np.fromfile(lidar_path, dtype=np.float32).reshape(-1,4)[:,:3]

    pcd = make_o3d_pointcloud(np.asarray(scene))
    return pcd

def main(args):
    dataset_path = args.data_path
    segment_name = args.segment
    segment_path = os.path.join(dataset_path, segment_name)
    
    lidar_paths = sorted(glob(os.path.join(segment_path, 'lidar', '*.bin')))
    text_features = text_extractor()

    point2image_mapper = PointCloudToImageMapper(
        image_dim=(800, 450),
        cut_bound=0)
    
    
    for lidar_path in tqdm(lidar_paths):
        pcd = scene_loader(lidar_path)
        n_points = len(pcd.points)
        scene_name = lidar_path.split('/')[-1].split('.')[0]
        
        vis_id = torch.zeros((n_points, 6), dtype=int, device=device)
        color = torch.zeros((n_points, 3), dtype=int, device=device)
        counter = torch.zeros((n_points), device=device)
        semseg = torch.zeros((len(NUSCENES_PREDEFINED), n_points), device=device)
        for img_id, camera_name in enumerate(camera_names):
            intr = np.load(os.path.join(dataset_path, segment_name, 'K', f'{camera_name}.npy'))
            pose = np.loadtxt(os.path.join(dataset_path, segment_name, 'pose', f'{scene_name}_{camera_name}.txt'))
            image = imageio.v3.imread(os.path.join(dataset_path, segment_name, 'color', f'{scene_name}_{camera_name}.jpg'))

            mapping = np.ones([n_points, 4], dtype=int)
            locs_in = np.asarray(pcd.points)

            mapping[:, 1:4] = point2image_mapper.compute_mapping(pose, locs_in, depth=None, intrinsic=intr)

            valid = mapping[:, 3]
            vis_id[:, img_id] = torch.from_numpy(valid)
            counter[valid!=0]+= 1

            color_temp = torch.tensor(image[mapping[:, 1],mapping[:, 2]], device=device)
            color[valid!=0] += color_temp[valid!=0]

            clip_path = os.path.join(dataset_path, segment_name, 'clip', f'{scene_name}_{camera_name}.npy')
            mask_path = os.path.join(dataset_path, segment_name, 'mask', f'{scene_name}_{camera_name}.npz')

            masks = sparse.load_npz(mask_path).todense()

            maskwise_clip_features = torch.tensor(np.load(clip_path), dtype=torch.float32).to(device)

            class_preds = (100.0 * maskwise_clip_features @ text_features.T).softmax(dim=-1)
            select_cls = torch.zeros_like(class_preds)

            max_scores, select_mask = torch.max(class_preds, dim=0)
            
            granularity = 0.5
            if granularity < 1:
                thr_scores = max_scores * granularity
                select_mask = []
                for i, thr in enumerate(thr_scores):
                    cls_pred = class_preds[:,i]
                    locs = torch.where(cls_pred > thr)
                    select_mask.extend(locs[0].tolist())
            for idx in select_mask:
                select_cls[idx] = class_preds[idx]

            mask_to_points = torch.tensor(masks[:,mapping[:, 1],mapping[:, 2]], dtype=torch.float32).to(device)

            semseg += torch.einsum("qc,qh->ch", select_cls.float(), mask_to_points)
  
        semseg = semseg[:,counter!=0]
        semseg = semseg.cpu().detach().numpy().T

        points = np.array(pcd.points)
        new_points = points[np.array(counter.cpu()!=0), :]
        pcd.points = o3d.utility.Vector3dVector(new_points)

        for text_idx, _ in enumerate(text_features):

            os.makedirs(f'./visualization/nuScenes/{segment_name}/output/{NUSCENES_PREDEFINED[text_idx]}', exist_ok=True)

            cmap = plt.get_cmap('jet')

            #rgba = cmap(semseg[:, text_idx])
            rgba = cmap((semseg[:, text_idx] - semseg[:, text_idx].min()) / (1 - semseg[:, text_idx].min())) # RGBA: [0,1]
            rgb = rgba[:, :3]

            pcd.colors = o3d.utility.Vector3dVector(rgb)
            o3d.io.write_point_cloud(f'./visualization/nuScenes/{segment_name}/output/{NUSCENES_PREDEFINED[text_idx]}/{scene_name}.pcd', pcd)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/dataset/nuScene_lidarseg')
    parser.add_argument('--segment', type=str, default='scene-0001')
    parser.add_argument('--threshold', type=float, default=0.)

    args = parser.parse_args()
    main(args)
