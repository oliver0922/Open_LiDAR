import os
import sys
import argparse
import numpy as np
from time import time
from tqdm import tqdm
import random
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter
#import wandb

import torch
import torch.optim as optim

import torch.backends.cudnn as cudnn
from torch_scatter import scatter

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor

import clip

from collections import OrderedDict


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.disnet import DisNet


from data.dataloader import get_mink_generator
from util.metric import intersectionAndUnionGPU, AverageMeter, evaluate
from util.logger import Logger
from util.util import save_checkpoint
from warnings import simplefilter
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

    print(f'IoU : {iou_class}')
    print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    return mIoU, mAcc, allAcc

def map_nuscene(config, pred):
    valid_mask = torch.where(config.text_to_nuscene != 255)[0]

    output = scatter(pred[:,valid_mask], config.text_to_nuscene[valid_mask].long(), dim=1, reduce="sum")

    return output

def state_dict_remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict

def text_extractor(text_set, device, normalize_feature=True): 
    model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
    model = model.to(device)
    text_set = [f"a {c} in a scene" for c in text_set]
    text_set = clip.tokenize(text_set).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_set)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

def update_nuscene_mIoU(config, pred, labels, meter):
    prediction=torch.argmax(pred,dim=1)

    prediction = config.text_to_nuscene[prediction]

    intersection, union, target = intersectionAndUnionGPU(prediction, labels.squeeze(), K=16)
    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()

    meter['intersection'].update(intersection), meter['union'].update(union), meter['target'].update(target)

def prediction_model(config, data, text_feats, model, device):

    coords, feats, clip_feats, masks, labels = data["coords"], data["feats"].float(), data["clip_feats"].float(), data["masks"], data["labels"]
    
    coords, feats, clip_feats, masks, labels = coords.to(device), feats.to(device), clip_feats.to(device), masks.to(device), labels.to(device)

    coords[:,1:] /= config.voxel_size    

    clip_feats = (clip_feats/(clip_feats.norm(dim=-1, keepdim=True)+1e-5))

    sinput = SparseTensor(feats.cuda(non_blocking=True), coords.cuda(non_blocking=True))

    labels = labels.cuda(non_blocking=True)

    clip_prediction = clip_feats.float() @ text_feats.float().T

    out_feats = model(sinput)
    out_feats = (out_feats/(out_feats.norm(dim=-1, keepdim=True)+1e-5))

    distill_loss = (1 - torch.nn.CosineSimilarity()(out_feats[masks], clip_feats)).nanmean()

    distill_prediction = out_feats.float() @ text_feats.float().T

    ensemble_mask =  out_feats[masks].max(dim=-1)[0] < clip_prediction.max(dim=-1)[0]
    invalid = torch.range(0,masks.size(0)-1)[masks][~ensemble_mask]
    mask_0 = masks.clone()
    mask_0[invalid.long()] = False 

    feat_ensemble = out_feats.clone()
    feat_ensemble[mask_0] = clip_feats[ensemble_mask]

    ensemble_prediction = feat_ensemble.float() @ text_feats.float().T

    ensemble_loss = (1 - torch.nn.CosineSimilarity()(feat_ensemble[masks], clip_feats)).mean()

    return distill_loss, ensemble_loss, clip_prediction, distill_prediction, ensemble_prediction, labels, masks



def train(config, model, optimizer, epoch, train_dataloader, device):
    model.train()
    
    pbar = tqdm(train_dataloader, desc = "Training")

    torch.backends.cudnn.enabled = False

    for idx, data in enumerate(pbar):
        
        distill_loss, ensemble_loss, _, _, _, _, _ = prediction_model(config, data, config.text_feats, model, device)
        
        loss = distill_loss 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        config.writer.add_scalar('Loss/Distill_loss',distill_loss,idx+epoch*len(train_dataloader))
        config.writer.add_scalar('Loss/Ensemble_loss',ensemble_loss,idx+epoch*len(train_dataloader))

        pbar.set_postfix({'Distill_loss' : distill_loss.item()})



# def val(config, criterion, text_feats_org, val_dataloader, models, epoch, device, writer, sanity=False):
def val(config, model, epoch, val_dataloader, device, sanity=False):
    pbar = tqdm(val_dataloader, desc = "Validation")

    torch.backends.cudnn.enabled = False
    clip_pred = []
    distill_pred = []
    ensemble_pred = []
    labels_ = []
    labels_masked = []
    
    for idx, data in enumerate(pbar):
        with torch.no_grad():
            distill_loss, ensemble_loss, clip_prediction, distill_prediction, ensemble_prediction, labels, masks = prediction_model(config, data, config.text_feats, model, device)

            config.writer.add_scalar('Val_Loss/Distill_loss',distill_loss,idx+epoch*len(val_dataloader))
            config.writer.add_scalar('Val_Loss/Ensemble_loss',ensemble_loss,idx+epoch*len(val_dataloader))

            #clip_pred.append(config.text_to_nuscene[torch.argmax(clip_prediction,dim=1)])

            distill_pred.append(config.text_to_nuscene[torch.argmax(distill_prediction,dim=1)])

            #ensemble_pred.append(config.text_to_nuscene[torch.argmax(ensemble_prediction,dim=1)])

            labels_.append(labels)
            labels_masked.append(labels[masks])
            

            pbar.set_postfix({'Distill_loss' : distill_loss.item()})
        if sanity and idx > 2:
            break

    #clip_pred = torch.cat(clip_pred)
    distill_pred = torch.cat(distill_pred)
    #ensemble_pred = torch.cat(ensemble_pred)
    labels_ = torch.cat(labels_).squeeze()
    labels_masked = torch.cat(labels_masked).squeeze()

    print('----------------------------------------')
    print('Original CLIP feature results')
    #clip_iou = evaluate(clip_pred.cpu().numpy(), labels_masked.cpu().numpy(), dataset='waymo')
    print('----------------------------------------')
    print('Distillation CLIP feature results')
    distill_iou = evaluate(distill_pred.cpu().numpy(), labels_.cpu().numpy(), dataset='waymo')
    print('----------------------------------------')
    print('Ensenble CLIP feature results')
    #ensemble_iou = evaluate(ensemble_pred.cpu().numpy(), labels_.cpu().numpy(), dataset='waymo')
    print('----------------------------------------')


    return clip_iou, distill_iou, ensemble_iou

def main(config):
    worker_seed = 20212288

    np.random.seed(worker_seed)
    random.seed(worker_seed)

    torch.autograd.set_detect_anomaly(True)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    models = DisNet(config).cuda()

    optimizer = optim.AdamW(models.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = config.lr, pct_start=0.04, steps_per_epoch=10, 
                            anneal_strategy="cos", div_factor=10.0,final_div_factor=100.0, epochs=config.max_epochs)

    start_epoch = 0

    if config.resume:
        if os.path.isfile(config.resume):
            print("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            new_state_dict = state_dict_remove_module(checkpoint['state_dict'])
            models.load_state_dict(new_state_dict, strict=True)
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(config.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))

    train_dataloader, val_dataloader = get_mink_generator(config.nuScene_path, config.batch_size, config.workers, dataset='nuscenes')
    train_waymo_dataloader, val_waymo_dataloader = get_mink_generator(config.Waymo_path, config.batch_size, config.workers, dataset='waymo')

    text_set = ['barrier', 'barricade', 'bicycle', 'bus', 'car', 'bulldozer', 'excavator', 'concrete mixer', 'crane', 'dump truck',
                'motorcycle', 'person', 'pedestrian','traffic cone', 'trailer', 'semi trailer', 'cargo container', 'shipping container', 'freight container',
                'truck', 'road', 'curb', 'traffic island', 'traffic median', 'sidewalk', 'grass', 'grassland', 'lawn', 'meadow', 'turf', 'sod',
                'building', 'wall', 'pole', 'awning', 'tree', 'trunk', 'tree trunk', 'bush', 'shrub', 'plant', 'flower', 'woods']
    
    
    text_feats = text_extractor(text_set, device=device)
    config.text_feats = text_feats


    #text_to_nuscene = torch.tensor([0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 6, 7, 8, 8, 8, 8, 8, 9, 10, 11, 11, 11, 
    #                                12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15]).to(device)
    text_to_nuscene = torch.tensor([11, 11, 12, 3, 1, 4, 4, 4, 4, 4, 13, 7, 7, 11, 4, 4, 4, 4, 4, 2, 18, 17, 20, 20, 
                                    22, 15, 15, 15, 15, 15, 15, 14, 14, 14, 10, 16, 16, 16, 15, 15, 15, 15, 15]).to(device)-1


    sys.stdout = Logger(os.path.join(config.save_path, config.tag, 'log.txt'))

    config.writer = SummaryWriter(log_dir = os.path.join(config.save_path,'log_'+config.tag))

    model_save_path = os.path.join(config.save_path, config.tag)
    os.makedirs(model_save_path, exist_ok=True)

    val_miou_max = 0

    config.text_set = text_set
    config.text_to_nuscene = text_to_nuscene

    model_save_path = os.path.join(config.save_path, config.tag)
    os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(start_epoch, config.max_epochs):
        if epoch != 0:
            print('Epoch : ', epoch)

            train(config, models, optimizer, epoch, train_dataloader, device)
            
            scheduler.step()
            if epoch % config.val_duration == 0:
                _,_,val_results = val(config, models, epoch, val_waymo_dataloader, device, sanity=False)

        elif config.resume:
            _,_,val_results = val(config, models, epoch, val_waymo_dataloader, device, sanity=False)

        else: 
            _,_,val_results = val(config, models, epoch, val_waymo_dataloader, device, sanity=True)

        if val_miou_max < val_results : 
            is_best = True
            val_miou_max = val_results
        
        else:
            is_best = False
        
        save_checkpoint({
                'epoch': epoch,
                'state_dict': models.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, is_best, model_save_path , f'checkpoint_{epoch}.pt') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Learning config
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--voxel_size', type=float, default=0.05)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--gpu', default = "0", type=str)
    parser.add_argument('--val_duration', type=int, default=2)
    parser.add_argument('--arch_3d', type=str, default='MinkUNet18A')
    
    #CLIP config 
    parser.add_argument('--feature_2d_extractor', default = "openseg", type=str)
    parser.add_argument('--fcgf', action = 'store_true')

    #Dataset config
    parser.add_argument('--nuScene_path', default = '/dataset/nuScene_lidarseg_split', type=str)
    parser.add_argument('--Waymo_path', default = '/dataset/Waymo_split', type=str)
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