# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gin 
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from detectron2.structures import BitMasks

@gin.configurable
class Maskgenerator(object):
    
    def __init__(self, model_type, checkpoint_path, device,
                 points_per_side=32,
                 points_per_batch=64,
                 pred_iou_thresh=0.88,
                 stability_score_thresh=0.95,
                 stability_score_offset=1.0,
                 box_nms_thresh = 0.7,
                 crop_n_layers = 0,
                 crop_nms_thresh = 0.7,
                 crop_overlap_ratio = 512 / 1500,
                 crop_n_points_downscale_factor = 1,
                 point_grids = None, 
                 min_mask_region_area = 0,
                 output_mode = 'binary_mask'):
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)

        self.mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                        points_per_side = points_per_side,
                                                        points_per_batch = points_per_batch,
                                                        pred_iou_thresh = pred_iou_thresh,
                                                        stability_score_thresh = stability_score_thresh,
                                                        stability_score_offset = stability_score_offset,
                                                        box_nms_thresh = box_nms_thresh,
                                                        crop_n_layers = crop_n_layers,
                                                        crop_nms_thresh = crop_nms_thresh,
                                                        crop_overlap_ratio = crop_overlap_ratio,
                                                        crop_n_points_downscale_factor = crop_n_points_downscale_factor,
                                                        point_grids = point_grids,
                                                        min_mask_region_area = min_mask_region_area,
                                                        output_mode = output_mode)


    def generate_mask(self, image):
        masks = self.mask_generator.generate(image)
        return masks
    
def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    ax.imshow(img)

def expand_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    expand_ratio: float = 1.0,
    max_h: int = None,
    max_w: int = None,
):
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    w = w * expand_ratio
    h = h * expand_ratio
    box = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    if max_h is not None:
        box[1] = max(0, box[1])
        box[3] = min(max_h - 1, box[3])
    if max_w is not None:
        box[0] = max(0, box[0])
        box[2] = min(max_w - 1, box[2])
    return [int(b) for b in box]


def crop_with_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    bbox: torch.Tensor,
    fill: Tuple[float, float, float] = (0, 0, 0),
    expand_ratio: float = 1.0,
):
    l, t, r, b = expand_box(*bbox, expand_ratio)
    _, h, w = image.shape
    l = max(l, 0)
    t = max(t, 0)
    r = min(r, w)
    b = min(b, h)
    new_image = torch.cat(
        [image.new_full((1, b - t, r - l), fill_value=val) for val in fill]
    )
    mask_bool = mask.bool()
    return image[:, t:b, l:r] * mask[None, t:b, l:r] + (~ mask_bool[None, t:b, l:r]) * new_image, mask[None, t:b, l:r]


if __name__ == "__main__":
    import os 
    import cv2
    import open3d as o3d 
    import glob 
    from tqdm import tqdm
    import gin 
    import argparse
    from torch.nn import functional as F
    import open_clip
    import sparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', default = 0, type=int)

    gin.parse_config_file('/root/code/open_lidar/config/config.gin')
    config = parser.parse_args()

    if config.device_num > 3 : 
        dataset = "Waymo"
    else : 
        dataset = "nuScene_lidarseg"

    total_set = os.listdir(f"/dataset/{dataset}")
    
    process_set = total_set[(config.device_num%4)::4]

    generator = Maskgenerator()

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='./ovseg_clip_l_9a1909.pth')

    PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
    PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)

    for scene in tqdm(process_set):
    
        image_path = glob.glob(f"/dataset/{dataset}/{scene}/color/*.jpg")
        os.makedirs(f"/dataset/{dataset}/{scene}/mask", exist_ok= True)
        os.makedirs(f"/dataset/{dataset}/{scene}/clip", exist_ok= True)
        for img_path in tqdm(image_path):

            image = cv2.imread(img_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with torch.no_grad(), torch.cuda.amp.autocast():
                masks = generator.generate_mask(image)
            pred_masks = [masks[i]['segmentation'][None,:,:] for i in range(len(masks))]
            pred_masks = np.row_stack(pred_masks)

            masks = sparse.COO(pred_masks)
            sparse.save_npz(img_path.replace('color', 'mask').replace('jpg','npz'), masks, compressed=True)

            pred_masks = BitMasks(pred_masks)
            bboxes = pred_masks.get_bounding_boxes()

            mask_fill = [255.0 * c for c in PIXEL_MEAN]

            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            regions = []
            for bbox, mask in zip(bboxes, pred_masks):
                region, _ = crop_with_mask(
                    image,
                    mask,
                    bbox,
                    fill=mask_fill,
                )
                regions.append(region.unsqueeze(0))
            regions = [F.interpolate(r.to(torch.float), size=(224, 224), mode="bicubic") for r in regions]

            pixel_mean = torch.tensor(PIXEL_MEAN).reshape(1, -1, 1, 1)
            pixel_std = torch.tensor(PIXEL_STD).reshape(1, -1, 1, 1)
            imgs = [(r/255.0 - pixel_mean) / pixel_std for r in regions]
            imgs = torch.cat(imgs)

            img_batches = torch.split(imgs, 32, dim=0)

            with torch.no_grad(), torch.cuda.amp.autocast():
                clip_model.cuda()
                image_features = []
                for img_batch in img_batches:
                    image_feat = clip_model.encode_image(img_batch.cuda().half())
                    image_feat /= image_feat.norm(dim=-1, keepdim=True)
                    image_features.append(image_feat.detach())
                image_features = torch.cat(image_features, dim=0)
            
            np.save(img_path.replace('color', 'clip').replace('jpg','npy'), image_features.cpu().numpy())
        



