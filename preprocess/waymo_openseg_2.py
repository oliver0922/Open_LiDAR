import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils as fu
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

DATA_PATH = Path('/dataset/waymo/waymo_processed_data_v0_5_0')

def extract_openseg_img_feature(img, openseg_model, regional_pool=True):
    '''Extract per-pixel OpenSeg features.'''

    text_emb = tf.zeros([1, 1, 768])
    # load RGB image
    np_image_string = img

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
    
    with open('/dataset/waymo/waymo_infos_train_sampling_5.pkl', 'rb') as f:
        infos = pickle.load(f)
    
    for info in infos:
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        lidar_file = DATA_PATH / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file) # (N, 6): [x, y, z, intensity, elongation, NLZ_flag]