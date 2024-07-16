import os
import math
import numpy as np
from glob import glob
from tqdm import tqdm
import torch

import json
import argparse

import os

import clip

from matplotlib import cm
import open3d as o3d 
from PIL import Image

import cv2

if __name__ == '__main__':

    clip_data_path = '/dataset/kwonyoung/Open_LiDAR/dataset/nuScene_lidarseg_split'

    #scene_name = 'scene-0240' # woman
    #scene_name = 'scene-0982' # excavator
    #scene_name = 'scene-0018' # crowd
    scene_name = 'scene-0671' # jeep


    index = 10

    clip_data_path = os.path.join(clip_data_path, 'train', scene_name)

    clip_feat_files= os.path.join(clip_data_path, 'openseg', '1535487739147001.npz')


    clip_feat_files = sorted(glob(clip_feat_files))

    clip_feat_file = clip_feat_files[0]

    lidar_file = clip_feat_file.replace('openseg', 'lidar').replace('npz', 'bin')

    images = clip_feat_file.replace('openseg', 'color').replace('.npz', '*.jpg')
    images = sorted(glob(images))

    for i, image in enumerate(images):
        img = cv2.imread(image)
        cv2.imwrite(f"./{i}.jpg", img)

    #text_query = ['an excavator']

    text_query = ['Jeep']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    text = clip.tokenize(text_query).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        

    feat = np.load(clip_feat_file)

    feat, mask = feat['feat'], feat['mask_full']
    
    feat = torch.from_numpy(feat).to(device)
    mask = torch.from_numpy(mask).numpy()

    feat /= (torch.norm(feat, dim=1, keepdim=True) + 1e-5)
    text_features /= (torch.norm(text_features, dim=1, keepdim=True) + 1e-5)

    cosine_sim = feat @ text_features.T

    #cosine_sim = torch.softmax(cosine_sim, dim=1)

    cos_sim = np.zeros(mask.shape)

    cos_sim[mask] = cosine_sim.cpu().numpy().squeeze()

    cos_sim /= np.max(cos_sim)

    cmap = cm.get_cmap('viridis')
    colors = cmap(cos_sim)

    lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])

    o3d.io.write_point_cloud('test.ply', pcd)



