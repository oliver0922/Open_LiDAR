import math
import os
import torch
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from tensorflow import io

import cv2


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


class KITTI_CLIPExtractor:
    def __init__(self, args):
        self.args = args
        self.dataroot = args.dataroot

        self.img_list = sorted(glob(join(self.dataroot, 'image_2', '*.png')))
        self.pc_list = sorted(glob(join(self.dataroot, 'velodyne', '*.bin')))
        self.calib_list = sorted(glob(join(self.dataroot, 'calib', '*.txt')))

        self.img_dim = (1220, 370)

        self.mapper = PointCloudToImageMapper(image_dim=self.img_dim, cut_bound=5)

        self.openseg_model =  tf2.saved_model.load(args.openseg_model,
                    tags=[tf.saved_model.tag_constants.SERVING],)


    def __len__(self):
        return len(self.img_list)

    def load_calibration(self, calib_path):
        with open(calib_path, 'r') as f:
            lines = f.readlines()
        calib = {}

        for line in lines:
            if len(line.split(':', 1)) == 2:
                key, value = line.split(':', 1)[0], line.split(':', 1)[1]
                calib[key] = np.array([float(x) for x in value.split()]).reshape(3,-1)

        return calib
    
    
    def load_point_cloud(self, point_cloud_path):
        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)
        return point_cloud

    def read_bytes(self, path):
        '''Read bytes for OpenSeg model running.'''

        with io.gfile.GFile(path, 'rb') as f:
            file_bytes = f.read()
        return file_bytes

    def adjust_intrinsic(self, intrinsic, intrinsic_image_dim, image_dim):
        if intrinsic_image_dim == image_dim:
            return intrinsic
        resize_width = int(math.floor(
            image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
        intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
        intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
        # account for cropping here
        intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
        intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
        return intrinsic

    def extract_openseg_img_feature(self, img_path, openseg_model, img_size=None, regional_pool=True):
        '''Extract per-pixel OpenSeg features.'''

        text_emb = tf.zeros([1, 1, 768])
        # load RGB image
        # np_image_string = image.numpy().tobytes()
        np_image_string = self.read_bytes(img_path)
        # run OpenSeg

        results = openseg_model.signatures['serving_default'](
                inp_image_bytes= tf.convert_to_tensor(np_image_string),
                inp_text_emb=text_emb)
        img_info = results['image_info']
        crop_sz = [
            int(img_info[0, 0] * img_info[2, 0]),
            int(img_info[0, 1] * img_info[2, 1])
        ]
        if regional_pool:
            image_embedding_feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
        else:
            image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
        
        if img_size is not None:
            feat_2d = tf.cast(tf.image.resize_nearest_neighbor(
                image_embedding_feat, img_size, align_corners=True)[0], dtype=tf.float16).numpy()
        else:
            feat_2d = tf.cast(image_embedding_feat[[0]], dtype=tf.float16).numpy()
        
        del results
        del image_embedding_feat
        feat_2d = torch.from_numpy(feat_2d).permute(2, 0, 1)  # dtype=torch.float16
        return feat_2d

        #return clip_feature_3d, mapping # 3D feature, 2D mapping

def get_args():
    '''Command line arguments.'''
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on nuScenes.')
    parser.add_argument('--dataroot', type=str, help='Where is the base logging directory', default="/dataset/kitti/testing")
    parser.add_argument('--openseg_model', type=str, default='/root/code/openseg_exported_clip', help='Where is the exported OpenSeg model')
    parser.add_argument('--run_device', type=int, help='multiframe_number', default=0)

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import multiprocessing as mp
    from functools import partial
    
    args = get_args()
    print("Arguments:")
    print(args)


    extractor = KITTI_CLIPExtractor(args)

    total_num = extractor.__len__()

    def load_data(index):

        image_path = extractor.img_list[index]
        point_cloud_path = extractor.pc_list[index]
        calib_path = extractor.calib_list[index]

        point_cloud = extractor.load_point_cloud(point_cloud_path)
        calib = extractor.load_calibration(calib_path)

        intrinsic = calib['P2'].reshape(3, 4)[:, :3]
        image_size = cv2.imread(image_path).shape[:2][::-1]

        intrinsic = extractor.adjust_intrinsic(intrinsic, image_size, extractor.img_dim)

        Tr_velo_to_cam = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), [0, 0, 0, 1]))
        R0_velo_to_cam = np.eye(4)
        R0_velo_to_cam[:3, :3] = calib['R0_rect'].reshape(3, 3)
        Tr_velo_to_cam = np.dot(R0_velo_to_cam, Tr_velo_to_cam)
        
        Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)



        mapping = extractor.mapper.compute_mapping(camera_to_world=Tr_cam_to_velo, 
                                            coords = point_cloud[:, :3], depth=None, 
                                            intrinsic=intrinsic)
        
        clip_feature_2d = extractor.extract_openseg_img_feature(image_path, extractor.openseg_model, 
                                                                img_size=[extractor.img_dim[1], extractor.img_dim[0]])

        clip_feature_3d = clip_feature_2d[:, mapping[:, 0], mapping[:, 1]].permute(1, 0).half().numpy()
        
        mask = mapping[:, 2].astype(np.uint8)

        np.savez_compressed(join(extractor.dataroot, 'openseg', f'{str(index).zfill(6)}.npz'), feat = clip_feature_3d, mask_full = mask)

    
    for i in range (0, total_num):
        load_data(i)

