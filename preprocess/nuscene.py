import os
import math
import multiprocessing as mp
import numpy as np
import imageio
import cv2
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import nuscenes.utils.geometry_utils as geoutils
import sys
sys.path.append('.')

from preprocess.util import adjust_intrinsic, process_txt, quaternion_rotation_matrix

np.set_printoptions(suppress=True) 

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

def nusc_to_pose(rotation, translation):
    pose = np.eye(4) 
    pose[:3,:3], pose[:3,3] = quaternion_rotation_matrix(rotation), translation

    return pose

def process_one_sequence(image_num):
    '''process one sequence.'''

    sample = nusc.sample[image_num]
    lidar_token = sample["data"]["LIDAR_TOP"]

    scene_token = nusc.sample[image_num]['scene_token']
    scene = nusc.get('scene', scene_token)
    scene_name = scene['name']


    out_dir_lidar = os.path.join(out_dir, scene_name, 'lidar')
    out_dir_lidar_pose = os.path.join(out_dir, scene_name, 'lidar_pose')
    out_dir_label = os.path.join(out_dir, scene_name, 'label')
    out_dir_color = os.path.join(out_dir, scene_name, 'color')
    out_dir_pose = os.path.join(out_dir, scene_name, 'pose')
    out_dir_K = os.path.join(out_dir, scene_name, 'K')

    os.makedirs(out_dir_color, exist_ok=True)
    os.makedirs(out_dir_pose, exist_ok=True)
    os.makedirs(out_dir_K, exist_ok=True)
    os.makedirs(out_dir_label, exist_ok=True)
    os.makedirs(out_dir_lidar, exist_ok=True)
    os.makedirs(out_dir_lidar_pose, exist_ok=True)

    sd_rec_lidar = nusc.get('sample_data', sample['data']["LIDAR_TOP"])

    pose_record_lidar = nusc.get('ego_pose', sd_rec_lidar['ego_pose_token'])
    
    cs_record = nusc.get("calibrated_sensor", sd_rec_lidar["calibrated_sensor_token"])

    pcd_to_car = geoutils.transform_matrix(cs_record["translation"], Quaternion(cs_record["rotation"]))

    poserecord = nusc.get('ego_pose', sd_rec_lidar['ego_pose_token'])

    car_to_global = geoutils.transform_matrix(poserecord["translation"], Quaternion(poserecord["rotation"]))

    pcd_to_global = car_to_global @ pcd_to_car 

    lidar_pose = pcd_to_global 

    lidar_point = np.fromfile(nusc.get_sample_data(lidar_token)[0], dtype=np.float32).reshape(-1,5)[:,:4]
    label = np.fromfile(os.path.join('/dataset/nuscenes', nusc.get('lidarseg', lidar_token)['filename']), dtype=np.uint8)

    timestamp = sample['timestamp']

    lidar_point.tofile(os.path.join(out_dir_lidar, str(timestamp)+'.bin'))
    label.tofile(os.path.join(out_dir_label, str(timestamp)+'.label'))

    def return_pose(cam):

        cam_data = nusc.get('sample_data', sample['data'][cam])

        poserecord_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])

        car_image_to_global = geoutils.transform_matrix(poserecord_cam["translation"], Quaternion(poserecord_cam["rotation"]))

        cs_record_cam = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        cam_to_car_image = geoutils.transform_matrix(cs_record_cam["translation"], Quaternion(cs_record_cam["rotation"]))

        cam_to_global = car_image_to_global @ cam_to_car_image

        return np.linalg.inv(pcd_to_global) @ cam_to_global

    cam_path = {'CAM_FRONT' : nusc.get_sample_data(sample["data"]["CAM_FRONT"])[0], 
                    'CAM_FRONT_RIGHT' : nusc.get_sample_data(sample["data"]["CAM_FRONT_RIGHT"])[0], 
                    'CAM_BACK_RIGHT' : nusc.get_sample_data(sample["data"]['CAM_BACK_RIGHT'])[0],
                    'CAM_BACK' : nusc.get_sample_data(sample["data"]["CAM_BACK"])[0], 
                    'CAM_BACK_LEFT' : nusc.get_sample_data(sample["data"]["CAM_BACK_LEFT"])[0], 
                    'CAM_FRONT_LEFT' : nusc.get_sample_data(sample["data"]["CAM_FRONT_LEFT"])[0]}

    cam_intrinsics = {'CAM_FRONT' : nusc.get_sample_data(sample["data"]["CAM_FRONT"])[2], 
                    'CAM_FRONT_RIGHT' : nusc.get_sample_data(sample["data"]["CAM_FRONT_RIGHT"])[2], 
                    'CAM_BACK_RIGHT' : nusc.get_sample_data(sample["data"]['CAM_BACK_RIGHT'])[2],
                    'CAM_BACK' : nusc.get_sample_data(sample["data"]["CAM_BACK"])[2], 
                    'CAM_BACK_LEFT' : nusc.get_sample_data(sample["data"]["CAM_BACK_LEFT"])[2], 
                    'CAM_FRONT_LEFT' : nusc.get_sample_data(sample["data"]["CAM_FRONT_LEFT"])[2]}

    
    #lidar_pose = nusc_to_pose(pose_record_lidar['rotation'], pose_record_lidar['translation'])
    np.savetxt(os.path.join(out_dir_lidar_pose, str(timestamp)+'.txt'), lidar_pose, delimiter = ' ') 
    
    cam_locs = ['CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']
    
    for cam in cam_locs:
        #img_name = os.path.join(data_path, scene_name, 'frames', timestamp, cam, 'color_image.jpg')
        img = imageio.v3.imread(cam_path[cam])
        img = cv2.resize(img, img_size)
        imageio.imwrite(os.path.join(out_dir_color, str(timestamp) + '_' + cam + '.jpg'), img)

        np.savetxt(os.path.join(out_dir_pose, str(timestamp)+ '_' + cam +'.txt'), return_pose(cam), delimiter = ' ')
        K = adjust_intrinsic(cam_intrinsics[cam], intrinsic_image_dim=(1600, 900), image_dim=img_size)
        np.save(os.path.join(out_dir_K, cam+'.npy'), K)
    
    print(scene_name, ' done')

nusc = NuScenes(version='v1.0-trainval', dataroot='/dataset/nuscenes', verbose=True)

#! YOU NEED TO MODIFY THE FOLLOWING
#####################################
split = 'v1.0_trainval' # 'train' | 'val'
out_dir = '/dataset/nuScene_lidarseg' # downloaded original nuscenes data
#####################################

os.makedirs(out_dir, exist_ok=True)

img_size = (800, 450)

global count
global prev_scene

count = 0
prev_scene = None

process_one_sequence(0)

import pdb; pdb.set_trace()

p = mp.Pool(processes=mp.cpu_count())
p.map(process_one_sequence, range(len(nusc.sample)))
p.close()
p.join()
