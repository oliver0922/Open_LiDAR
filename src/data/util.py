import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torch

class SAM_mask_extractor(object):
    def __init__(model_path):

    def generate_mask(self, image):

class CLIP_feature_extractor(object):
    def __init__(model_path):

    def generate_clip_feature(self, mask):



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
    
def make_o3d_pointcloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.array(color))
    return pcd
    
def class_map(dataset):
    if dataset == 'nuscenes':
        CLASS_REMAP = 256*np.ones(32) # map from 32 classes to 16 classes
        CLASS_REMAP[2] = 7 # person
        CLASS_REMAP[3] = 7
        CLASS_REMAP[4] = 7
        CLASS_REMAP[6] = 7
        CLASS_REMAP[9] = 1 # barrier
        CLASS_REMAP[12] = 8 # traffic cone
        CLASS_REMAP[14] = 2 # bicycle
        CLASS_REMAP[15] = 3 # bus
        CLASS_REMAP[16] = 3
        CLASS_REMAP[17] = 4 # car
        CLASS_REMAP[18] = 5 # construction vehicle
        CLASS_REMAP[21] = 6 # motorcycle
        CLASS_REMAP[22] = 9 # trailer ???
        CLASS_REMAP[23] = 10 # truck
        CLASS_REMAP[24] = 11 # drivable surface
        CLASS_REMAP[25] = 12 # other flat??
        CLASS_REMAP[26] = 13 # sidewalk
        CLASS_REMAP[27] = 14 # terrain
        CLASS_REMAP[28] = 15 # manmade
        CLASS_REMAP[30] = 16 # vegetation

    elif dataset == 'waymo':
        CLASS_REMAP = np.array(range(23), dtype=np.uint8)
        CLASS_REMAP[0] = 256

    return CLASS_REMAP

def minkowski_collate_fn(batch):
    r"""
    Collation function for MinkowskiEngine.SparseTensor that creates batched
    cooordinates given a list of dictionaries.
    """
    coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(
        [d["coords"] for d in batch],
        [d["feats"] for d in batch],
        [d["labels"] for d in batch],
        dtype=torch.float32,
    )
    feats_3d_batch = torch.cat([d["feats_3d"] for d in batch], dim=0)
    mask_batch = torch.cat([d["mask"] for d in batch], dim=0)
    dataset_batch = [d["dataset"] for d in batch]

    return {
        "coords": coords_batch,
        "feats" : feats_batch,
        "feats_3d" : feats_3d_batch,
        "labels": labels_batch,
        "mask" : mask_batch,
        "dataset" : dataset_batch
    }

def normalize_intensity(raw_data):
    intensity = raw_data[:,3]
    intensity = (intensity-min(intensity))/(max(intensity)-min(intensity))-0.5
    raw_data[:,3] = intensity
    return raw_data

