import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import open3d as o3d


def heading2rotmat(heading_angle):
    rotmat = np.zeros((3,3))
    rotmat[2,2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
    return rotmat

def get_oriented_bbox(points):
    pca = PCA(2)
    pca.fit(points[:, :2])

    yaw_vec = pca.components_[0, :]
    yaw = np.arctan2(yaw_vec[1], yaw_vec[0])

    points_tmp = points.copy()
    points_tmp = heading2rotmat(-yaw) @ points_tmp[:, :3].T

    xyz_min = points_tmp.min(axis=1)
    xyz_max = points_tmp.max(axis=1)
    diff = xyz_max - xyz_min

    bbox = np.array([xyz_min, xyz_max]) @ heading2rotmat(yaw).T

    center = (bbox[0] + bbox[1]) / 2
    cx, cy, cz = center
    bbox_values = np.expand_dims(np.array([cx, cy, cz, diff[0] / 2, diff[1] / 2, diff[2] / 2, -yaw]), axis=0)

    return bbox_values

def get_dbscan_clusters(points):
    clustering = DBSCAN(eps=0.5, min_samples=10).fit(points)
    labels = clustering.labels_
    return labels

def get_label_points(points, target_label):
    return points[points[:, -1] == target_label]

def get_bboxes_from_label_points(points, target_label):
    label_points = get_label_points(points, target_label)
    clusters = get_dbscan_clusters(label_points[:, :3])

    bboxes = []
    for cluster in np.unique(clusters):
        cluster_points = label_points[clusters == cluster]
        bboxes.append(get_oriented_bbox(cluster_points))

    return bboxes

