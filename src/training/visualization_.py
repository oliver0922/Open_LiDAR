import os
import sys
import argparse
import numpy as np
from time import time
from tqdm import tqdm
import random
import open3d as o3d
#import wandb

import torch
import torch.optim as optim

import torch.backends.cudnn as cudnn
from torch_scatter import scatter

import MinkowskiEngine as ME
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# from dataset.nuscenes import LiDAR_feature_dataset, minkowski_collate_fn

from data.dataloader import get_generator
from util.metric import intersectionAndUnionGPU, AverageMeter
from util.logger import Logger
from warnings import simplefilter
from training.prediction_sam import PredictionwithMapping


import matplotlib.pyplot as plt
simplefilter(action='ignore', category=FutureWarning)


def worker_init_fn(worker_id):
    '''Worker initialization.'''
    random.seed(time.time() + worker_id)

def return_results(meter):
    intersection_meter = meter['intersection']
    union_meter = meter['union']
    target_meter = meter['target']
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    template = {''}
    print(f'IoU : {iou_class}')
    print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    return mIoU, mAcc, allAcc

def map_nuscene(config, pred):
    valid_mask = torch.where(config.text_to_nuscene != 255)[0]

    output = scatter(pred[:,valid_mask], config.text_to_nuscene[valid_mask].long(), dim=1, reduce="sum")

    return output



# def val(config, criterion, text_feats_org, val_dataloader, models, epoch, device, writer, sanity=False):
def val(config, prediction_module, val_dataloader, sanity=True):
    pbar = tqdm(val_dataloader, desc = "Validation")

    torch.backends.cudnn.enabled = False
    CLIP_meter = {'intersection' : AverageMeter(), 'union': AverageMeter(), 'target': AverageMeter()}
    
    for idx, data in enumerate(pbar):
        with torch.no_grad():
            semseg, labels = prediction_module.predict(data)
            
            pred = map_nuscene(config, semseg)

            intersection, union, target = intersectionAndUnionGPU(torch.argmax(pred,dim=1), labels, K=16)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            CLIP_meter['intersection'].update(intersection), CLIP_meter['union'].update(union), CLIP_meter['target'].update(target)

        if sanity and idx > 3000:
            break

    print('----------------------------------------')
    print('Original LiDAR/CLIP feature results')
    CLIP_results = return_results(CLIP_meter)

    return CLIP_results

def make_o3d_pointcloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.array(color))
    return pcd

def visualization_sequences(config, prediction_module, val_dataloader, sanity=True):
    pbar = tqdm(val_dataloader, desc = "Validation")

    torch.backends.cudnn.enabled = False

    for idx, data in enumerate(pbar):
        if idx < 40:
            continue 

        with torch.no_grad():
            semseg, labels, masks = prediction_module.predict(data)

            #import pdb; pdb.set_trace()

            for text_idx, text_query in enumerate(prediction_module.text_set):

                os.makedirs(f'./visualization/nuScenes/{config.tag}/{text_query}', exist_ok=True)

                cmap = plt.get_cmap('jet')

                #rgba = cmap(semseg[:, text_idx])
                rgba = cmap(semseg[:, text_idx]) # RGBA: [0,1]
                rgb = rgba[:, :3]

                pcd = make_o3d_pointcloud(data['coords'].squeeze()[masks].cpu().numpy(), rgb)
                
                o3d.io.write_point_cloud(f'./visualization/nuScenes/{config.tag}/{text_query}/{str(idx).zfill(6)}.pcd', pcd)

        

        if sanity and idx > 80:
            break


def main(config):
    worker_seed = 20212288

    np.random.seed(worker_seed)
    random.seed(worker_seed)

    torch.autograd.set_detect_anomaly(True)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    _, val_dataloader = get_generator(config.nuScene_path, config.batch_size, config.workers, dataset='nuscenes')

    
    text_set = ['barrier','barricade','bicycle','bus','car','bulldozer','excavator','concrete mixer',
                'crane', 'dump truck', 'motorcycle', 'pedestrian','person','traffic cone','trailer','semi trailer',
                'cargo container', 'shipping container', 'freight container', 'truck','road','curb', 'traffic island',
                'traffic median','sidewalk','grass','grassland','lawn','meadow','turf','sod','building','wall','pole',
                'awning','tree','trunk','tree trunk','bush','shrub','plant','flower','woods']
                #'bicyclist','motorcyclist', 'traffic sign', 'traffic light', 'parking space', 'parking lot', 'road stud', 'speed bump', 'railtrack', 'lane marker']
    

    #nuScene classes
    #text_set = ['barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle', 'pedestrian', 'traffic cone', 'trailer', 'truck', 'driveable surface', 'other flat','sidewalk', 'terrain', 'manmade', 'vegetation']

    #SemanticKITTI
    #text_set = ['car', 'bicycle', 'motorcycle', 'truck', 'other vehicle', 'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other ground','building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']

    #Waymo classes
    #text_set = ['car', 'truck', 'bus', 'motorcyclist', 'bicyclist', 'pedestrian', 'sign', 'traffic light', 'pole', 'construction cone', 'bicycle', 'motorcycle',
    # 'building', 'vegetation', 'tree trunk', 'curb', 'road', 'lane marker', 'walkable', 'sidewalk', 'other ground', 'other vehicle']



    prediction_module = PredictionwithMapping(text_set, device)
    '''
    text_to_nuscene = 256*torch.ones(len(text_set),)
    text_to_nuscene[[1,2]] = 1
    text_to_nuscene[[3,44]] = 2
    text_to_nuscene[[4]] = 3
    text_to_nuscene[[5]] = 4
    text_to_nuscene[[6,7,8,9,10]] = 5
    text_to_nuscene[[11,45]] = 6
    text_to_nuscene[[12,13]] = 7
    text_to_nuscene[[14]] = 8
    text_to_nuscene[[15,16,17,18,19]] = 9
    text_to_nuscene[[20]] = 10
    text_to_nuscene[[21]] = 11
    text_to_nuscene[[22,23,24,48,49]] = 12
    text_to_nuscene[[25]] = 13
    text_to_nuscene[[26,27,28,29,30,31]] = 14
    text_to_nuscene[[32,33,34,35]] = 15
    text_to_nuscene[[36,37,38,39,40,41,42,43]] = 16
    text_to_nuscene -= 1 

    config.text_to_nuscene = text_to_nuscene

    gt_nusc = 256*torch.ones(32, )
    gt_nusc[[8]] = 1
    gt_nusc[[13]] = 2
    gt_nusc[[14,15]] = 3
    gt_nusc[[16]] = 4
    gt_nusc[[17]] = 5
    gt_nusc[[20]] = 6
    gt_nusc[[1,2,3,5]] = 7
    gt_nusc[[11]] = 8
    gt_nusc[[21]] = 9
    gt_nusc[[22]] = 10
    gt_nusc[[23]] = 11
    gt_nusc[[24]] = 12
    gt_nusc[[25]] = 13
    gt_nusc[[26]] = 14
    gt_nusc[[27]] = 15
    gt_nusc[[29]] = 16
    gt_nusc -= 1

    config.gt_nusc = gt_nusc
    '''
    sys.stdout = Logger(os.path.join(config.save_path, config.tag, 'log.txt'))
    #writer = SummaryWriter(log_dir = os.path.join(config.save_path,'log_'+config.tag))

    model_save_path = os.path.join(config.save_path, config.tag)
    os.makedirs(model_save_path, exist_ok=True)

    visualization_sequences(config, prediction_module, val_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Learning config
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--voxel_size', type=float, default=0.05)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--gpu', default = "0", type=str)
    parser.add_argument('--val_duration', type=int, default=1)
    parser.add_argument('--arch_3d', type=str, default='MinkUNet18A')
    
    #CLIP config 
    parser.add_argument('--feature_2d_extractor', default = "openseg", type=str)
    parser.add_argument('--Distillation', action = 'store_true')
    parser.add_argument('--CLIP_projection', action = 'store_true')
    parser.add_argument('--LiDAR_projection', action = 'store_true')
    parser.add_argument('--projection', action = 'store_true')
    parser.add_argument('--prompt_eng', action = 'store_true')
    parser.add_argument('--fcgf', action = 'store_true')
    parser.add_argument('--projection_dim', type=int, default=32)
    parser.add_argument('--projection_num', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--loss', default = "cos", type=str)
    parser.add_argument('--cls_loss', default = "ce", type=str)
    
    #Dataset config
    parser.add_argument('--nuScene_path', default = '/dataset/nuScene_lidarseg_split', type=str)
    parser.add_argument('--Waymo_path', default = '/home/jinmo/Data/Waymo', type=str)
    parser.add_argument('--SemanticKITTI', action = 'store_true')
    parser.add_argument('--nuScene_lidarseg', action = 'store_true')
    parser.add_argument('--Waymo_prediction', action = 'store_true')

    #Augmentation config 
    parser.add_argument('--dataset_mix', action = 'store_true')

    #model save config
    parser.add_argument('--tag', type=str)
    parser.add_argument('--save_path', default = "/root/code/open_lidar", type=str)
    parser.add_argument('--resume', default = None, type=str)
    parser.add_argument('--CLIP_proj_resume', default = None, type=str)

    config = parser.parse_args()
    main(config)