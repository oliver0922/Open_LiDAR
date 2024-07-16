import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils as fu
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
#from waymo_camera_ops.py_camera_model_ops import world_to_image

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
from tqdm import tqdm

world_to_image = py_camera_model_ops.world_to_image


def extract_openseg_img_feature(img, openseg_model, regional_pool=True):
    '''Extract per-pixel OpenSeg features.'''

    text_emb = tf.zeros([1, 1, 768])
    # load RGB image
    np_image_string = img
    #np_image_string = bytearray(img)
    # np_image_string = self.read_bytes(img_path)
    # run OpenSeg


    import pdb; pdb.set_trace()

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
    img_size = tf.image.decode_jpeg(img).numpy().shape[0:2]
    img_size = [img_size[0], img_size[1]]
    feat_2d = tf.cast(tf.image.resize_nearest_neighbor(
                image_embedding_feat, img_size, align_corners=True)[0], dtype=tf.float16).numpy()
    
    del results
    del image_embedding_feat
    feat_2d = np.array(torch.from_numpy(feat_2d).permute(2, 0, 1))  # dtype=torch.float16
    return feat_2d

def get_camera_projection(save_path, frame):
    ris, cps, _, pps = fu.parse_range_image_and_camera_projection(frame)
    P, cp = fu.convert_range_image_to_point_cloud(frame, ris, cps, pps, 0)
    P = P[0]
    cp = cp[0]

    count = np.zeros((len(P), 1), dtype=np.int8)
    feat_3d = np.zeros((len(P), 768), dtype=np.float16)

    for image in frame.images:
        #camera = index+1 
        img = image.image
        camera = image.name 
        
        feat_2d = extract_openseg_img_feature(img, openseg_model, regional_pool=True)
        
        valid_idx = np.where(cp[..., 0] == camera)[0]

        feat_3d[valid_idx, :] += feat_2d[:,cp[valid_idx][:,2], cp[valid_idx][:,1]].T
        count[valid_idx] += 1
        
    feat_3d /= (count + 1e-5)
    mask = count > 0

    np.savez_compressed(save_path, feat = feat_3d, mask_full = mask)


    



if __name__ == '__main__':

    waymo_root = '/dataset/waymo/'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    frame = dataset_pb2.Frame()
    openseg_model =  tf2.saved_model.load('/root/code/openseg_exported_clip', tags=[tf.saved_model.tag_constants.SERVING],)

    tf_record_list = sorted(glob(join(waymo_root, 'training', '*.tfrecord')))

    idx = 0

    f = open('./train_split.txt', 'r')
    lines = f.readlines()
    tf_record_list = []

    for line in lines:
        tf_record_list.append(line.split('\n')[0])


    for tf_record in tqdm(tf_record_list):
        tf_record_path = os.path.join(waymo_root, 'training', tf_record)
        dir_path = os.path.join(waymo_root, 'training_openseg', tf_record.split('.')[0])
        os.makedirs(dir_path, exist_ok=True)
        ds = tf.data.TFRecordDataset(tf_record_path, compression_type='')
        idx_tfrecord = 0
        for data in ds:
            if idx % 5 == 0:
                frame.ParseFromString(data.numpy())
                import pdb; pdb.set_trace()
                save_path = os.path.join(dir_path, f'{str(idx_tfrecord).zfill(6)}.npz')
                get_camera_projection(save_path, frame)
            idx_tfrecord += 1
            idx += 1