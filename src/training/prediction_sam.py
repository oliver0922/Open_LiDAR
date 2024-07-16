import torch
import numpy as np
import open_clip
import clip

from util.mapper import PointCloudToImageMapper
#from sort.sort import *

import tensorflow as tf2
import tensorflow.compat.v1 as tf
from util.fusion_util import extract_openseg_img_feature

def text_extractor(text_set, device, normalize_feature=True): 
    model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
    model = model.to(device)
    text_set = [f"a {c} in a scene" for c in text_set]
    text_set = clip.tokenize(text_set).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_set)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


def text_extractor_ovseg(text_set, device, normalize_feature=True): 
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='/root/code/open_lidar/ovseg_clip_l_9a1909.pth')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    text_inputs = torch.cat([tokenizer(f"a photo of a {c}") if c != 'others' else tokenizer(c) for c in text_set]).to(device)

    text_features = model.encode_text(text_inputs)
    text_features /= (text_features.norm(dim=-1, keepdim=True)+1e-10)

    return text_features

class PredictionwithMapping():
    def __init__(self, text_set, device, granularity = 0.9, dim=768):

        self.text_features = text_extractor(text_set, device)     
   
        self.text_features_ovseg = text_extractor_ovseg(text_set, device)

        self.text_set = text_set
        self.mapper = PointCloudToImageMapper(image_dim=(800, 450), cut_bound=5)
        self.device = device
        self.dim = dim
        self.granularity = granularity
        self.openseg_model = tf2.saved_model.load('/root/code/open_lidar/openseg', tags=[tf.saved_model.tag_constants.SERVING],)
        self.text_emb = tf.zeros([1, 1, dim])


    def predict(self, metadata):

        points, img_paths, intrs, poses, maskwise_clips, masks, label = metadata['coords'], metadata['colors_path'], metadata['intrinsics'], metadata['poses'], metadata['clips'], metadata['masks'], metadata['labels']
        assert len(intrs) == len(poses) == len(maskwise_clips) == len(masks)

        num_img = len(intrs)
        

        mask_entire = (label!=255).squeeze()

        locs_in = points.squeeze()

        locs_in = locs_in[mask_entire]

        n_points = len(locs_in)

        pointwise_feature = torch.zeros((n_points, self.dim), device=self.device)

        counter = torch.zeros((n_points,1), device=self.device)
        semseg = torch.zeros((n_points, len(self.text_set)), device=self.device)

        vis_id = torch.zeros((n_points, num_img), dtype=int, device=self.device)

        
        img_id = 0
        for intr, img_dir, pose, maskwise_clip, masks_per_img in zip(intrs, img_paths, poses, maskwise_clips, masks):

            if len(masks_per_img)==0:
                continue 

            mapping = np.ones([n_points, 4], dtype=int)

            mapping[:, 1:4] = self.mapper.compute_mapping(pose.squeeze(), locs_in, depth=None, intrinsic=intr.squeeze())
            if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
                continue
            mapping = torch.from_numpy(mapping).to(self.device)

            mask = mapping[:, 3]
            vis_id[:, int(img_id)] = mask
            counter[mask!=0]+= 1
            '''
            # 1. Openseg features
            img_feature = extract_openseg_img_feature(img_dir[0], self.openseg_model, self.text_emb, img_size=[450, 800]).to(self.device)

            point_feature = img_feature[:,mapping[:,1], mapping[:,2]].permute(1, 0)

            point_feature /= (point_feature.norm(dim=-1, keepdim=True)+1e-5)

            pointwise_feature[mask!=0,:] += point_feature[mask!=0,:]

            del img_feature

            class_preds = (100.0 * point_feature.half() @ self.text_features.T).softmax(dim=-1)
            #semseg += class_preds
            
            
            max_scores, select_point = torch.max(class_preds, dim=0)
            
            mask_to_points = torch.tensor(masks_per_img.squeeze()[:,mapping[:, 1],mapping[:, 2]], dtype=torch.float32).to(self.device)

            mean_preds_per_mask = torch.einsum("pc,mp->cm", class_preds.float(), mask_to_points) / (mask_to_points.sum(dim=1)+1e-5)

            max_mean_score_per_mask, select_mask = mean_preds_per_mask.max(0)


            select_cls_per_mask = torch.zeros_like(mean_preds_per_mask)
            
            if self.granularity < 1:
                thr_scores = max_mean_score_per_mask * self.granularity
                select_mask = []
                for i, thr in enumerate(thr_scores):
                    cls_pred = mean_preds_per_mask[:,i]
                    locs = torch.where(cls_pred > thr)
                    select_mask.extend(locs[0].tolist())
            for idx in select_mask:
                select_cls_per_mask[idx] = mean_preds_per_mask[idx] 

            select_cls_by_mask = (select_cls_per_mask @ mask_to_points).T
            
            pixelwise_select_cls = torch.zeros_like(class_preds)
            
            if self.granularity < 1:
                thr_scores = max_scores * self.granularity
                select_point = []
                for i, thr in enumerate(thr_scores):
                    cls_pred = class_preds[:,i]
                    locs = torch.where(cls_pred > thr)
                    select_point.extend(locs[0].tolist())
            for idx in select_point:
                pixelwise_select_cls[idx] = class_preds[idx]

            '''

            #positive_mask = select_cls_by_mask > select_cls
            #negative_mask = select_cls_by_mask == 0

            #select_cls[positive_mask] = select_cls_by_mask.half()[positive_mask]
            #select_cls[negative_mask] = 0
            
            #3 OV-Seg
            
            class_preds = (100.0 * maskwise_clip.to(self.device).squeeze() @ self.text_features_ovseg.T).softmax(dim=-1)
            select_cls = torch.zeros_like(class_preds)

            max_scores, select_mask = torch.max(class_preds, dim=0)
            
            if self.granularity < 1:
                thr_scores = max_scores * self.granularity
                select_mask = []
                for i, thr in enumerate(thr_scores):
                    cls_pred = class_preds[:,i]
                    locs = torch.where(cls_pred > thr)
                    select_mask.extend(locs[0].tolist())
            for idx in select_mask:
                select_cls[idx] = class_preds[idx]

            mask_to_points = torch.tensor(masks_per_img.squeeze()[:,mapping[:, 1],mapping[:, 2]], dtype=torch.float32).to(self.device)

            #pointwise_feature += torch.einsum("qc,qh->ch", maskwise_clip.to(self.device).squeeze(), mask_to_points)

            maskwise_select_cls = torch.einsum("qc,qh->ch", select_cls.float(), mask_to_points)

            semseg += maskwise_select_cls.T
            
            #tendency = pixelwise_select_cls/(maskwise_select_cls.T+1e-5)

            #total_select_cls = pixelwise_select_cls

            #total_select_cls[tendency > 1] = maskwise_select_cls.half().T[tendency > 1]

            #semseg += total_select_cls

            img_id +=1
        counter[counter==0] = 1e-5

        pointwise_feature /= counter

        semseg /= counter

        pointwise_feature /= (pointwise_feature.norm(dim=-1, keepdim=True)+1e-10)

        class_preds = 100*(pointwise_feature.half() @ self.text_features.T).softmax(dim=-1)

        del pointwise_feature

        max_scores, select_point = torch.max(class_preds, dim=0)


        point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

        mask = torch.zeros(n_points, dtype=torch.bool)
        mask[point_ids] = True

        mask_entire[mask_entire==True] = mask

        #return class_preds[mask].detach().cpu().numpy(), label[0][mask_entire], mask_entire
        return semseg[mask].detach().cpu(), label[0][mask_entire], mask_entire


class PredictionwithAssociation():
    def __init__(self, text_set, device, granularity = 0.7, dim=768):

        self.text_features = text_extractor(text_set, device)     
   
        self.text_set = text_set
        self.mapper = PointCloudToImageMapper(image_dim=(800, 450), cut_bound=0)
        self.device = device
        self.dim = dim
        self.granularity = granularity

    def predict(self, metadata):

        points, intrs, poses, maskwise_clips, masks, label = metadata['coords'], metadata['intrinsics'], metadata['poses'], metadata['clips'], metadata['masks'], metadata['labels']
        assert len(intrs) == len(poses) == len(maskwise_clips) == len(masks)


        n_points = len(points.squeeze())

        counter = torch.zeros((n_points), device=self.device)
        semseg = torch.zeros((len(self.text_set), n_points), device=self.device)

        locs_in = points.squeeze()

        pointwise_feature = torch.zeros((768, n_points), device=self.device)


        for intr, pose, maskwise_clip, masks_per_img in zip(intrs, poses, maskwise_clips, masks):

            mapping = np.ones([n_points, 4], dtype=int)

            mapping[:, 1:4] = self.mapper.compute_mapping(pose.squeeze(), locs_in, depth=None, intrinsic=intr.squeeze())

            valid = mapping[:, 3]
            counter[valid!=0]+= 1

            class_preds = (100.0 * maskwise_clip.to(self.device).squeeze() @ self.text_features.T).softmax(dim=-1)
            select_cls = torch.zeros_like(class_preds)

            max_scores, select_mask = torch.max(class_preds, dim=0)
            
            if self.granularity < 1:
                thr_scores = max_scores * self.granularity
                select_mask = []
                for i, thr in enumerate(thr_scores):
                    cls_pred = class_preds[:,i]
                    locs = torch.where(cls_pred > thr)
                    select_mask.extend(locs[0].tolist())
            for idx in select_mask:
                select_cls[idx] = class_preds[idx]

            mask_to_points = torch.tensor(masks_per_img.squeeze()[:,mapping[:, 1],mapping[:, 2]], dtype=torch.float32).to(self.device)

            pointwise_feature += torch.einsum("qc,qh->ch", maskwise_clip.to(self.device).squeeze(), mask_to_points)

            semseg += torch.einsum("qc,qh->ch", select_cls.float(), mask_to_points)

        counter[counter==0] += 1e-5

        pointwise_feature /= counter

        semseg = semseg[:,counter!=0]
        semseg = semseg.cpu().detach().T
        label = label.squeeze()[counter!=0]

        return semseg, label
    
if __name__ == "__main__":
    train_dataset = LiDARDataset(dataset='nuscenes', dataset_dir='/dataset/nuScene_lidarseg_split', split='train')
    data = train_dataset[0]