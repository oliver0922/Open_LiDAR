import tensorflow as tf
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import multiprocessing
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils as fu
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

world_to_image = py_camera_model_ops.world_to_image

import torch

def extract_openseg_img_feature_batch(imgs, openseg_model, regional_pool=True):
    '''Extract per-pixel OpenSeg features in batch.'''
    text_emb = tf.zeros([len(imgs), 1, 768])
    
    import pdb; pdb.set_trace()
    np_image_strings = [tf.convert_to_tensor(img) for img in imgs]
    import pdb; pdb.set_trace()


    results = openseg_model.signatures['serving_default'](inp_image_bytes=np_image_strings,inp_text_emb=text_emb)
    
    import pdb; pdb.set_trace()


    img_info = results['image_info']
    crop_sz = [
        int(img_info[0, 0] * img_info[2, 0]),
        int(img_info[0, 1] * img_info[2, 1])
    ]
    
    if regional_pool:
        image_embedding_feats = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
    else:
        image_embedding_feats = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
    
    feats_2d = []
    for i in range(len(imgs)):
        img_size = tf.image.decode_jpeg(imgs[i]).numpy().shape[0:2]
        img_size = [img_size[0], img_size[1]]
        feat_2d = tf.cast(tf.image.resize(
            image_embedding_feats[i], img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        ), dtype=tf.float16).numpy()
        
        feat_2d = np.array(torch.from_numpy(feat_2d).permute(2, 0, 1))
        feats_2d.append(feat_2d)
    
    return feats_2d

def get_camera_projection(save_path, frame, openseg_model):
    
    ris, cps, _, pps = fu.parse_range_image_and_camera_projection(frame)
    P, cp = fu.convert_range_image_to_point_cloud(frame, ris, cps, pps, 0)
    P = P[0]
    cp = cp[0]

    count = np.zeros((len(P), 1), dtype=np.int8)
    feat_3d = np.zeros((len(P), 768), dtype=np.float16)

    imgs = [image.image for image in frame.images]
    feats_2d = extract_openseg_img_feature_batch(imgs, openseg_model, regional_pool=True)

    for i, image in enumerate(frame.images):
        camera = image.name
        valid_idx = np.where(cp[..., 0] == camera)[0]
        
        feat_3d[valid_idx, :] += feats_2d[i][:, cp[valid_idx][:, 2], cp[valid_idx][:, 1]].T
        count[valid_idx] += 1
        
    feat_3d /= (count + 1e-5)
    mask = count > 0

    np.savez_compressed(save_path, feat=feat_3d, mask_full=mask)

def process_tf_record(tf_record, waymo_root, openseg_model_path, idx, step=5):
    frame = dataset_pb2.Frame()
    openseg_model = tf.saved_model.load(openseg_model_path)
    tf_record_path = os.path.join(waymo_root, 'training', tf_record)
    dir_path = os.path.join(waymo_root, 'training_openseg', tf_record.split('.')[0])
    os.makedirs(dir_path, exist_ok=True)
    ds = tf.data.TFRecordDataset(tf_record_path, compression_type='')

    
    idx_tfrecord = idx % step
    for data in ds:
        if idx % step == 0:
            frame.ParseFromString(data.numpy())
            save_path = os.path.join(dir_path, f'{str(idx_tfrecord).zfill(6)}.npz')
            get_camera_projection(save_path, frame, openseg_model)
        idx += 1
        idx_tfrecord += 1

    return idx


if __name__ == '__main__':
    waymo_root = '/dataset/waymo/'
    openseg_model_path = '/root/code/openseg_exported_clip'
    
    with open('./train_split.txt', 'r') as f:
        tf_record_list = [line.strip() for line in f.readlines()]

    idx = 0
    frame = dataset_pb2.Frame()
    for tf_record in tqdm(tf_record_list):
        idx = process_tf_record(tf_record, waymo_root, openseg_model_path, idx)