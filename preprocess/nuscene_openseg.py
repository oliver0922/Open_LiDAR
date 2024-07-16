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

import imageio
import cv2

from nuscenes.nuscenes import NuScenes
import nuscenes.utils.geometry_utils as geoutils
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarSegPointCloud, LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes

NUSCENES_FULL_CLASSES = ( # 32 classes
    'noise',
    'animal',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier',
    'movable_object.debris',
    'movable_object.pushable_pullable',
    'movable_object.trafficcone',
    'static_object.bicycle_rack',
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck',
    'flat.driveable_surface',
    'flat.other',
    'flat.sidewalk',
    'flat.terrain',
    'static.manmade',
    'static.other',
    'static.vegetation',
    'vehicle.ego',
    'unlabeled',
)

VALID_NUSCENES_CLASS_IDS = ()
dynamic_label = [1,2,3,4,5,6,7,8,9,10]
NUSCENES_CLASS_REMAP = 256*np.ones(32) # map from 32 classes to 16 classes
NUSCENES_CLASS_REMAP[2] = 7 # person
NUSCENES_CLASS_REMAP[3] = 7
NUSCENES_CLASS_REMAP[4] = 7
NUSCENES_CLASS_REMAP[6] = 7
NUSCENES_CLASS_REMAP[9] = 1 # barrier
NUSCENES_CLASS_REMAP[12] = 8 # traffic cone
NUSCENES_CLASS_REMAP[14] = 2 # bicycle
NUSCENES_CLASS_REMAP[15] = 3 # bus
NUSCENES_CLASS_REMAP[16] = 3
NUSCENES_CLASS_REMAP[17] = 4 # car
NUSCENES_CLASS_REMAP[18] = 5 # construction vehicle
NUSCENES_CLASS_REMAP[21] = 6 # motorcycle
NUSCENES_CLASS_REMAP[22] = 9 # trailer ???
NUSCENES_CLASS_REMAP[23] = 10 # truck
NUSCENES_CLASS_REMAP[24] = 11 # drivable surface
NUSCENES_CLASS_REMAP[25] = 12 # other flat??
NUSCENES_CLASS_REMAP[26] = 13 # sidewalk
NUSCENES_CLASS_REMAP[27] = 14 # terrain
NUSCENES_CLASS_REMAP[28] = 15 # manmade
NUSCENES_CLASS_REMAP[30] = 16 # vegetation

color_map = {255 : np.array([0, 0, 0]),
0 : np.array([0, 0, 255]),
  1: np.array([245, 150, 100]),
  2: np.array([245, 230, 100]),
  3: np.array([250, 80, 100]),
  4: np.array([150, 60, 30]),
  5: np.array([255, 0, 0]),
  6: np.array([180, 30, 80]),
  7: np.array([255, 0, 0]),
  8: np.array([30, 30, 255]),
  9: np.array([200, 40, 255]),
  10: np.array([90, 30, 150]),
  11: np.array([255, 0, 255]),
  12: np.array([255, 150, 255]),
  13: np.array([75, 0, 75]),
  14: np.array([75, 0, 175]),
  15: np.array([0, 200, 255]),
  16 : np.array([0, 0, 255]),
  17: np.array([245, 150, 100]),
  18: np.array([245, 230, 100]),
  19: np.array([250, 80, 100]),
  20: np.array([150, 60, 30]),
  21: np.array([255, 0, 0]),
  22: np.array([180, 30, 80]),
  23: np.array([255, 0, 0]),
  24: np.array([30, 30, 255]),
  25: np.array([200, 40, 255]),
  26: np.array([90, 30, 150]),
  27: np.array([255, 0, 255]),
  28: np.array([255, 150, 255]),
  29: np.array([75, 0, 75]),
  30: np.array([75, 0, 175]),
  31: np.array([0, 200, 255])}




def get_args():
    '''Command line arguments.'''
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on nuScenes.')
    parser.add_argument('--dataroot', type=str, help='Where is the base logging directory', default="/dataset/nuscenes")
    parser.add_argument('--frame_num', type=int, help='multiframe_number', default=5)
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory', default="/dataset/nuScene_lidarseg_split")
    parser.add_argument('--openseg_model', type=str, default='./CLIP/exported_model/', help='Where is the exported OpenSeg model')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--run_device', type=int, help='multiframe_number', default=0)

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
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


def return_cam(nusc, sample, cam):

    cam_data = nusc.get('sample_data', sample['data'][cam])

    poserecord_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])

    car_image_to_global = geoutils.transform_matrix(poserecord_cam["translation"], Quaternion(poserecord_cam["rotation"]))

    cs_record_cam = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    cam_to_car_image = geoutils.transform_matrix(cs_record_cam["translation"], Quaternion(cs_record_cam["rotation"]))

    cam_to_global = car_image_to_global @ cam_to_car_image

    cam_path = nusc.get_sample_data(sample["data"][cam])[0]

    cam_intrinsic = nusc.get_sample_data(sample["data"][cam])[2]

    return (cam_to_global, cam_path, cam_intrinsic)

def inner_points_bbox(ann, pcd):

    pcd_transform = (np.linalg.inv(ann.rotation_matrix)@(pcd-ann.center).T).T
    #pcd_transform = (ann.rotation_matrix@(pcd-ann.center).T).T

    ann_max_bound = (ann.wlh/2)[[1,0,2]]
    ann_min_bound = (-ann.wlh/2)[[1,0,2]]

    mask = np.logical_and(pcd_transform < ann_max_bound, pcd_transform > ann_min_bound)
    mask = np.logical_and(mask[:,0],np.logical_and(mask[:,1], mask[:,2]))

    return mask

def load_frame(dataroot, nusc, sample, mask=None, return_lidarpose=True):
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd_rec_lidar = nusc.get('sample_data', lidar_token)

    cs_record = nusc.get("calibrated_sensor", sd_rec_lidar["calibrated_sensor_token"])
    pcd_to_car = geoutils.transform_matrix(cs_record["translation"], Quaternion(cs_record["rotation"]))
    poserecord = nusc.get('ego_pose', sd_rec_lidar['ego_pose_token'])

    car_to_global = geoutils.transform_matrix(poserecord["translation"], Quaternion(poserecord["rotation"]))
    pcd_to_global = car_to_global @ pcd_to_car 
    lidar_pose = pcd_to_global 

    lidar_path = nusc.get_sample_data(lidar_token)[0]
    is_key_frame = nusc.get('sample_data', lidar_token)['is_key_frame']

    label_path = nusc.get('lidarseg', lidar_token)['filename']

    labels = np.fromfile(os.path.join(dataroot,label_path), dtype=np.uint8)
   
    coords = np.fromfile(lidar_path, dtype=np.float32).reshape(-1,5)[:,:4]
    lidar_h = coords
    lidar_h[:,3] = 1
    coords[:,:3] = (lidar_pose @ lidar_h.T).T[:,:3]

    if mask is not None:
        coords, labels = coords[mask,:], labels[mask]

    if return_lidarpose:
        return coords[:,:3], labels, lidar_pose
    else:
        return coords[:,:3], labels




def export_dynamic_frames(dataroot, nusc, sample, out_dir, frame_num = 5, single_frame=False):

    coords_total = []
    labels_total = []
    
    coords_std, labels_std, pose_std = load_frame(dataroot, nusc, sample, mask=None, return_lidarpose=True)
    coords_total.append(coords_std)
    labels_total.append(labels_std)

    anns = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])[1]

    #next_sample = sample
    prev_sample = sample

    sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sample_rec = nusc.get('sample', sd_record['sample_token'])
    chan = sd_record['channel']
    ref_chan = 'LIDAR_TOP'

    pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=frame_num)

    #cam_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']


    #img_list = []
    #for cam in cam_list:
    #    img_list.append(return_cam(nusc, sample, cam))

    #mask = []
    #mask.append(np.ones_like(labels_std))
    '''
    for i in range(sample_num):
        
        #next_flag = next_sample['next'] !=''
        prev_flag = prev_sample['prev'] !=''
        
        if next_flag:
            #import pdb; pdb.set_trace()
            next_sample = nusc.get('sample',next_sample['next'])
            next_sample_data_token = next_sample['data']['LIDAR_TOP']
            next_lidar_path = nusc.get_sample_data(next_sample_data_token)[0]
            next_pcd = LidarSegPointCloud(next_lidar_path)
            next_label= np.fromfile(os.path.join(dataroot, nusc.get('lidarseg', next_sample_data_token)['filename']), dtype=np.int8)
            next_static_mask = np.ones_like(next_label)

            for cam in cam_list:
                img_list.append(return_cam(nusc, next_sample, cam))
        
        if prev_flag:
            prev_sample = nusc.get('sample',prev_sample['prev'])
            prev_sample_data_token = prev_sample['data']['LIDAR_TOP']
            prev_lidar_path = nusc.get_sample_data(prev_sample_data_token)[0]
            prev_pcd = LidarSegPointCloud(prev_lidar_path)
            prev_label= np.fromfile(os.path.join(dataroot, nusc.get('lidarseg', prev_sample_data_token)['filename']), dtype=np.int8)
            prev_static_mask = np.ones_like(prev_label)

            for cam in cam_list:
                img_list.append(return_cam(nusc, prev_sample, cam))

        ####Dynamic_objects_aggregation####
        if not single_frame:
            for obj in anns:
                orientation = obj.orientation
                center = obj.center
                std = nusc.get('sample_annotation', obj.token)
                std_position = np.eye(4)
                std_position[:3,:3] = obj.rotation_matrix
                std_position[:3,3] = obj.center 

                std = nusc.get('sample_annotation', obj.token)
                next_std = std
                prev_std = std

              
                ######Next######
                if next_std['next'] != '':
                    next_std = nusc.get('sample_annotation', next_std['next'])

                    ann_next = nusc.get_sample_data(next_sample_data_token, selected_anntokens=[next_std['token']], use_flat_vehicle_coordinates=False)[1][0]

                    ann_next_position = np.eye(4)
                    ann_next_position[:3, :3] = ann_next.rotation_matrix 
                    ann_next_position[:3, 3] = ann_next.center

                    next_mask = inner_points_bbox(ann_next, next_pcd.points[:,:3])
                    next_static_mask = np.logical_and(next_static_mask, ~next_mask)

                    if len(np.where(next_mask)[0]) > 0:

                        sample_pcd_next, sample_label_next = next_pcd.points[next_mask,:3], next_label[next_mask] 
                        sample_pcd_next = np.concatenate((sample_pcd_next,np.ones((sample_pcd_next.shape[0],1))), axis=1)
                        sample_pcd_next = (pose_std @ std_position @ np.linalg.inv(ann_next_position) @ sample_pcd_next.T).T[:,:3]
                        coords_total.append(sample_pcd_next)
                        labels_total.append(sample_label_next)
                '''
                ######Prev######
    '''
                if prev_std['prev'] != '':

                    prev_std = nusc.get('sample_annotation', prev_std['prev'])

                    ann_prev = nusc.get_sample_data(prev_sample_data_token, selected_anntokens=[prev_std['token']], use_flat_vehicle_coordinates=False)[1][0]

                    ann_prev_position = np.eye(4)
                    ann_prev_position[:3, :3] = ann_prev.rotation_matrix 
                    ann_prev_position[:3, 3] = ann_prev.center
                
                    prev_mask = inner_points_bbox(ann_prev, prev_pcd.points[:,:3])

                    prev_static_mask = np.logical_and(prev_static_mask, ~prev_mask)

                    if len(np.where(prev_mask)[0]) > 0:

                        sample_pcd_prev, sample_label_prev = prev_pcd.points[prev_mask,:3], prev_label[prev_mask] 
                        sample_pcd_prev = np.concatenate((sample_pcd_prev,np.ones((sample_pcd_prev.shape[0],1))), axis=1)
                        sample_pcd_prev = (pose_std @ std_position @ np.linalg.inv(ann_prev_position) @ sample_pcd_prev.T).T[:,:3]
                        coords_total.append(sample_pcd_prev)
                        labels_total.append(sample_label_prev)
                        mask.append(np.zeros_like(sample_label_prev))
                '''
            ####Static_background_aggregation####
    '''
            if next_flag:
                coords_next, labels_next = load_frame(dataroot, nusc, next_sample, mask=next_static_mask, return_lidarpose=False)
                coords_total.append(coords_next)
                labels_total.append(labels_next)
            
            if prev_flag:
                coords_prev, labels_prev, pose_prev= load_frame(dataroot, nusc, prev_sample, mask=None, return_lidarpose=True)
                coords_total.append(coords_prev)
                labels_total.append(labels_prev)
                mask.append(np.zeros_like(labels_prev))
            '''
    #coords_total = np.concatenate(coords_total, axis=0)
    
    coords_total = pc.points.T[:,:3]
    #labels_total = np.ones_like(coords_total[:,0])*255
    #labels_total[:len(labels_std)] = labels_std
    #labels_total = np.concatenate(labels_total, axis=0)
    ##coords_total_h = np.concatenate([coords_total, np.ones_like(labels_total.reshape(-1,1))], axis=1)
                
    #coords_total = (np.linalg.inv(pose_std) @ coords_total_h.T).T[:,:3]

    #mask_total = np.concatenate(mask, axis=0)


    ####debug####
    '''
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()

    v3d = o3d.utility.Vector3dVector
    cloud.points = v3d(coords_total[:,:3])
    colors = np.zeros_like(coords_total[:,:3])
    for i,label in enumerate(labels_total):
        colors[i,:] = color_map[label][[2,1,0]]/255
    cloud.colors = v3d(colors)

    o3d.io.write_point_cloud('debug.pcd', cloud)
    '''
    return coords_total

def process_one_scene(image_num, args):
    '''Process one scene.'''

    # short hand

    # load 3D data (point cloud, color and the corresponding labels)
    # Only process points with GT label annotation

    splits = create_splits_scenes()

    


    nusc = args.nusc
    out_dir = args.output_dir
    
    sample = nusc.sample[image_num]

    timestamp = sample['timestamp']
    coords_total = export_dynamic_frames(args.dataroot, nusc, sample, out_dir, frame_num = args.frame_num, single_frame=False)
    scene_name = nusc.get('scene', sample["scene_token"])['name']
    if args.frame_num!=5:
        if scene_name in splits['train']:
            out_dir_multi_lidar = os.path.join(out_dir, 'train', scene_name, f'multi_lidar_{args.frame_num}')

        elif scene_name in splits['val']:
            out_dir_multi_lidar = os.path.join(out_dir, 'val', scene_name, f'multi_lidar_{args.frame_num}')
    else : 
        if scene_name in splits['train']:
            out_dir_multi_lidar = os.path.join(out_dir, 'train', scene_name, f'multi_lidar')

        elif scene_name in splits['val']:
            out_dir_multi_lidar = os.path.join(out_dir, 'val', scene_name, f'multi_lidar')
    
    os.makedirs(out_dir_multi_lidar, exist_ok=True)
    coords_total.tofile(join(out_dir_multi_lidar,f'{timestamp}.bin'))
    #mask_total.tofile(join(out_dir,nusc.get('scene', sample["scene_token"])['name'],f'{timestamp}.npy'))

    print(nusc.get('scene', sample["scene_token"])['name'], image_num)


def process_one_scene_clip(image_num,  args):
    '''Process one scene.'''
    
    nusc = args.nusc
    out_dir = args.out_dir
    # short hand
    img_size = args.img_dim
    point2img_mapper = args.point2img_mapper
    openseg_model = args.openseg_model
    text_emb = args.text_emb
    #cam_locs = ['back', 'back_left', 'back_right', 'front', 'front_left', 'front_right']
    cam_locs = ['CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']

    # load 3D data (point cloud, color and the corresponding labels)
    # Only process points with GT label annotation

    timestamp = sample['timestamp']
    coords_total, _, mask_total = export_dynamic_frames(args.dataroot, nusc, sample, frame_num = args.frame_num, single_frame=True)

    locs_in = torch.from_numpy(aggregated_data[0])
    labels_in = torch.from_numpy(aggregated_data[2])
    mask_entire = labels_in!=255

    mask_static = labels_in >= 10
    mask_dynamic = labels_in < 10 

    locs_in = locs_in[mask_entire]
    n_points = locs_in.shape[0]
    
    #if exists(join(out_dir, f'{timestamp}.pt')):
    #    print(f'{timestamp}.pt' + ' already done!')
    #    return 1
    

    # process 2D features
    device = torch.device('cpu')

    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, args.feat_dim), device=device)

    num_img = len(img_list)

    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)

    pbar = tqdm(img_list, desc="images")

    for img_id, data in enumerate(pbar):
    
        intr = adjust_intrinsic(data[2], intrinsic_image_dim=(1600, 900), image_dim=img_size)
        pose = data[0]
        img = imageio.v3.imread(data[1])
        img = cv2.resize(img, img_size)
        imageio.imwrite(f'./temp_{str(args.run_device)}.jpg', img)
        img_dir = f'./temp_{str(args.run_device)}.jpg'
        # calculate the 3d-2d mapping

        feat_2d = extract_openseg_img_feature(
            img_dir, openseg_model, text_emb, img_size=[img_size[1], img_size[0]]).to(device)

        mapping = np.ones([n_points_cur, 4], dtype=int)
        
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth=None, intrinsic=intr)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask
        
        feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)

        counter[mask!=0 & mask_static]+= 1
        sum_features[mask!=0 & mask_static] += feat_2d_3d[mask!=0 & mask_static]
        if img_id < len(cam_locs):
            counter[mask!=0 & mask_dynamic]+= 1
            sum_features[mask!=0 & mask_dynamic] += feat_2d_3d[mask!=0 & mask_dynamic]
        pbar.set_postfix({'mask_len' : len(torch.where(mask!=0)[0])})
    
    counter[counter==0] = 1e-5
    feat_bank = sum_features/counter
    point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

    mask = torch.zeros(n_points, dtype=torch.bool)
    mask[point_ids] = True
    mask_entire[mask_entire==True] = mask
    feat_bank_cpu = feat_bank[mask].half().cpu()

    torch.save({"feat": feat_bank_cpu,
                "mask_full": mask_entire},
            join(out_dir, f'{timestamp}.pt'))

    print(join(out_dir, f'{timestamp}.pt') + ' is saved!')

if __name__ == "__main__":
    import multiprocessing as mp
    from functools import partial
    
    args = get_args()
    print("Arguments:")
    print(args)

    nusc = NuScenes(version='v1.0-trainval', dataroot='/dataset/nuscenes', verbose=True)
    #sample = nusc.sample[10]

    #export_dynamic_frames('/dataset/nuScenes_lidarseg', nusc, sample, frame_num = 1)

    args.nusc = nusc

    total_num = len(nusc.sample)

    id_range = None

    p = mp.Pool(processes=mp.cpu_count())
    p.map(partial(process_one_scene, args=args), range(total_num))
    p.close()
    p.join()
