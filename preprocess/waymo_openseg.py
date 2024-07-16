import os
import math
import numpy as np
import imageio
from glob import glob
from tqdm import tqdm
import tensorflow.compat.v1 as tf
import torch
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import json
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
tf.config.experimental.enable_tensor_float_32_execution(enabled=False)
np.set_printoptions(suppress=True) 

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    """Adjusts the intrinsic matrix to the image size.

    Args:
        intrinsic: original intrinsic matrix
        intrinsic_image_dim: original image size
        image_dim: target image size

    Returns:
        4x4 adjusted intrinsic matrix
    """
    if intrinsic_image_dim == image_dim:
        return intrinsic
    
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))

    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)

    return intrinsic

def convert_range_image_to_point_cloud_labels(frame, range_images, segmentation_labels, ri_index=0):
    """Convert segmentation labels from range images to point clouds.

    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
        segmentation_labels: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)
        
        point_labels.append(sl_points_tensor.numpy())
    return point_labels

def make_dirs(out_dir, split, scene_name, targets):
    """Make output directories.

    Args:
        out_dir: output directory for preprocessed dataset
        split: split name (e.g. 'training', 'validation')
        scene_name: scene name (e.g. 'scene-xxxx')
        targets: target data types (e.g. ['color', 'lidar', 'label', 'lidar_pose', 'pose', 'K', 'bbox'])

    Returns:
        out_dirs: paths to output directories, keys: ['color', 'lidar', 'label', 'lidar_pose', 'pose', 'K', 'bbox']
    """
    targets = ['color', 'lidar', 'label', 'lidar_pose', 'pose', 'K', 'bbox']
    out_dirs = {}

    if split == 'training':
        split = 'train'
    elif split == 'validation':
        split = 'val'

    for target in targets:
        out_dirs[target] = os.path.join(out_dir, split, scene_name, target)
        os.makedirs(os.path.join(out_dir, split, scene_name, target), exist_ok=True)
    return out_dirs


def read_bytes(path):
    '''Read bytes for OpenSeg model running.'''

    with io.gfile.GFile(path, 'rb') as f:
        file_bytes = f.read()
    return file_bytes

def extract_openseg_img_feature(image, openseg_model, img_size=None, regional_pool=True):
    '''Extract per-pixel OpenSeg features.'''

    text_emb = tf.zeros([1, 1, 768])
    # load RGB image
    # np_image_string = image.numpy().tobytes()
    #np_image_string = read_bytes(img_path)
    np_image_string = image.tobytes()
    
    
    # run OpenSeg

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
    
    if img_size is not None:
        feat_2d = tf.cast(tf.image.resize_nearest_neighbor(
            image_embedding_feat, img_size, align_corners=True)[0], dtype=tf.float16).numpy()
    else:
        feat_2d = tf.cast(image_embedding_feat[[0]], dtype=tf.float16).numpy()
    
    del results
    del image_embedding_feat
    feat_2d = torch.from_numpy(feat_2d).permute(2, 0, 1)  # dtype=torch.float16
    return feat_2d



def process_one_sequence(args, split, scene_path, scene_name):
    """Process one sequence of Waymo Open Dataset.

    Args:
        scene_path: path to the sequence (e.g. '/dataset/Waymo/training/segment-1001709016809349153_5725_000_5745_000_with_camera_labels.tfrecord')
        scene_name: processed scene name (e.g. 'scene-xxxx')
    """
    img_size = (800, 450)
    dataset = tf.data.TFRecordDataset(scene_path, compression_type='')

    out_dir = args.out_dir
    out_dirs = make_dirs(out_dir, split, scene_name, args.targets)

    cam_names = {1: 'CAM_FRONT', 2: 'CAM_FRONT_LEFT', 3: 'CAM_FRONT_RIGHT', 4: 'CAM_SIDE_LEFT', 5: 'CAM_SIDE_RIGHT'}

    axes_transformation = np.array([[0, -1,  0, 0],
                                    [0,  0, -1, 0],
                                    [1,  0,  0, 0],
                                    [0,  0,  0, 1]])
    
    for data in dataset:
        frame = open_dataset.Frame()

        frame.ParseFromString(bytearray(data.numpy()))
        calibrations = frame.context.camera_calibrations

        # Camera intrinsics
        if 'K' in args.targets:
            for calibration in calibrations:
                name = calibration.name
                width = calibration.width
                height = calibration.height

                K = np.eye(3)
                K[0, 0], K[1, 1], K[0, 2], K[1, 2] = calibration.intrinsic[:4]
                
                K = adjust_intrinsic(K, intrinsic_image_dim=(width, height), image_dim=img_size)
                #np.save(os.path.join(out_dirs['K'], f'{cam_names[name]}.npy'), K)

        if frame.lasers[0].ri_return1.segmentation_label_compressed:
            (range_images, camera_projections, segmentation_labels, range_image_top_pose) = \
                frame_utils.parse_range_image_and_camera_projection(frame)

            # LiDAR points and 3D labels
            if 'lidar' in args.targets or 'label' in args.targets:
                points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)
                points_all = np.concatenate(points, axis=0)

                point_labels = convert_range_image_to_point_cloud_labels(frame, range_images, segmentation_labels)
                point_labels_all = np.concatenate(point_labels, axis=0)

                points_labels_concat = np.hstack((points_all, point_labels_all))
                points_labels_concat_filtered = points_labels_concat[points_labels_concat[:, 3] != 0]

                assert torch.from_numpy(points_labels_concat_filtered[:, :3]).size(1) == 3

                if 'lidar' in args.targets:
                    points_labels_concat_filtered[:, :3].tofile(os.path.join(out_dirs['lidar'], f'{str(frame.timestamp_micros)}.bin'))
                    
                if 'label' in args.targets:
                    points_labels_concat_filtered[:, 4].astype(np.uint8).tofile(os.path.join(out_dirs['label'], f'{str(frame.timestamp_micros)}.label'))

            # Pose of LiDAR(top)
            if 'lidar_pose' in args.targets:
                lidar_pose = np.asarray(frame.images[0].pose.transform).reshape(4, 4) @ np.asarray(frame.context.laser_calibrations[0].extrinsic.transform).reshape(4, 4)
                np.savetxt(os.path.join(out_dirs['lidar_pose'], f'{str(frame.timestamp_micros)}.txt'), lidar_pose)

            # Camera images and poses
            if 'color' in args.targets or 'pose' in args.targets:
                for idx, image in enumerate(frame.images):
                    name = image.name
                    #if 'color' in args.targets:
                    image_jpeg = tf.image.decode_jpeg(image.image)
                    image_jpeg = np.asarray(tf.image.resize(image_jpeg, (img_size[1], img_size[0]))).astype(np.uint8)
                    
                        imageio.imwrite(os.path.join(out_dirs['color'], f'{str(frame.timestamp_micros)}_{cam_names[name]}.jpg'), image_jpeg)

                    if 'pose' in args.targets:
                        cam_extrinsic = np.asarray(frame.context.camera_calibrations[name - 1].extrinsic.transform).reshape(4, 4)
                        pose = np.asarray(image.pose.transform).reshape(4, 4) @ cam_extrinsic @ np.linalg.inv(axes_transformation)
                        np.savetxt(os.path.join(out_dirs['pose'], f'{str(frame.timestamp_micros)}_{cam_names[name]}.txt'), pose)
            
            if 'bbox' in args.targets:
                type_map = {0: 'TYPE_UNKNOWN', 1: 'TYPE_VEHICLE', 2: 'TYPE_PEDESTRIAN', 3: 'TYPE_SIGN', 4: 'TYPE_CYCLIST'}
                labels = frame.laser_labels

                label_list = []
                for label in labels:
                    bbox_dict = {}

                    # TYPE_UNKNOWN
                    if label.type == 0:
                        continue

                    # bounding box type (e.g. TYPE_VEHICLE)
                    bbox_dict['type'] = type_map[label.type]

                    # bounding box information
                    bbox_dict['box'] = {}
                    bbox_dict['box']['center_x'] = label.box.center_x
                    bbox_dict['box']['center_y'] = label.box.center_y
                    bbox_dict['box']['center_z'] = label.box.center_z

                    bbox_dict['box']['length'] = label.box.length
                    bbox_dict['box']['width'] = label.box.width
                    bbox_dict['box']['height'] = label.box.height

                    bbox_dict['box']['heading'] = label.box.heading
                    
                    # bounding box metadata
                    bbox_dict['metadata'] = {}

                    bbox_dict['metadata']['speed_x'] = label.metadata.speed_x
                    bbox_dict['metadata']['speed_y'] = label.metadata.speed_y
                    bbox_dict['metadata']['speed_z'] = label.metadata.speed_z

                    # bounding box acceleration
                    bbox_dict['metadata']['accel_x'] = label.metadata.accel_x
                    bbox_dict['metadata']['accel_y'] = label.metadata.accel_y
                    bbox_dict['metadata']['accel_z'] = label.metadata.accel_z

                    # bounding box detection difficulty level
                    bbox_dict['detection_difficulty_level'] = label.detection_difficulty_level
                    bbox_dict['tracking_difficulty_level'] = label.tracking_difficulty_level
                    
                    # bounding box tracking id that is unique for each object track
                    bbox_dict['id'] = label.id

                    label_list.append(bbox_dict)

                with open(os.path.join(out_dirs['bbox'], f'{str(frame.timestamp_micros)}.json'), 'w') as f:
                    json.dump(label_list, f, indent=4)


def main(args):
    dataset_path = args.dataset_path
    splits = args.splits   

    if not set(splits) <= set(['training', 'validation']):
        raise ValueError('Invalid split name.')
    if not set(args.targets) <= set(['color', 'lidar', 'label', 'lidar_pose', 'pose', 'K', 'bbox']):
        raise ValueError('Invalid target name.')

    for split in splits:
        print(f'Processing {split} split')
        scene_list = sorted(glob(f'{dataset_path}/{split}/*'))
        pbar = tqdm(scene_list)

        if args.end is None:
            args.end = len(scene_list)
        range = (args.start, args.end)

        for i, scene in enumerate(pbar):
            if i < range[0] or i >= range[1]:
                continue
            scene_name = f'scene-{str(i).zfill(4)}' if split == 'training' else f'scene-{str(i + 798).zfill(4)}'
            pbar.set_description(f"Processing {scene.split('/')[-1].split('.')[0][8:45]}")
            process_one_sequence(args, split, scene, scene_name)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waymo Open Dataset Preprocessing')
    parser.add_argument('--dataset_path', type=str, default='/home/jinmo/Data/kwonyoung/waymo')
    parser.add_argument('--splits', type=str, nargs='+', default=['training', 'validation'])
    parser.add_argument('--out_dir', type=str, default='/home/jinmo/Data/jinmo/Waymo_split')
    parser.add_argument('--targets', type=str, nargs='+', default=['color', 'lidar', 'label', 'lidar_pose', 'pose', 'K', 'bbox'])
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()
    main(args)
