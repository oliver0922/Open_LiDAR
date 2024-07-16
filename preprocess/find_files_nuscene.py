from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.splits import create_splits_scenes
import os
import pickle
import matplotlib.pyplot as plt


# nuScenes 데이터셋 로드
nusc = NuScenes(version='v1.0-trainval', dataroot='/dataset/kwonyoung/nuScenes', verbose=True)

cam_channel = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK_RIGHT','CAM_BACK_LEFT','CAM_BACK']

#cam_name = 'n015-2018-09-25-11-10-38+0800__CAM_FRONT_RIGHT__1537845410620339.jpg'
cam_name = 'n008-2018-08-28-16-16-48-0400__CAM_BACK_RIGHT__1535487739128113.jpg'

for i,sample in enumerate(nusc.sample):
    # sample 데이터에서 camera_channel에 해당하는 데이터 가져오기
    for channel in cam_channel:
        sample_data_token = sample['data'][channel]
        sample_data = nusc.get('sample_data', sample_data_token)
    
    # sample_data의 파일 이름이 주어진 파일 이름과 일치하는지 확인
        if os.path.basename(sample_data['filename']) == cam_name:
            scene_name = nusc.get('scene', sample['scene_token'])['name']
            print(scene_name)

