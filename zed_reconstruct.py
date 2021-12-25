"""
Usage: 
python zed_reconstruct.py --scene=lab_dataset/ --mode=1
"""
import numpy as np
import cv2
import argparse
import os
import open3d as o3d
from PIL import Image
import time
import copy
import pandas as pd
import math
import sys
from sklearn.neighbors import NearestNeighbors


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mode", required=True, type=int, help='0: ground truth, 1:icp, 2: gt+icp')

    return parser.parse_args()

def draw_registration_result(source, target, transformation, paint_uniform_color=True):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if paint_uniform_color:
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def depth_image_to_point_cloud(rgb_img, depth_img, intrinsic_mtx):
    rgb = cv2.imread(rgb_img)[:,:,[2,1,0]]
    rgb = rgb[:,200:,:].reshape((-1,3))
    depth = cv2.imread(depth_img, cv2.IMREAD_UNCHANGED)
    depth_scale = 1000000.0
    fx, fy, cx, cy = intrinsic_mtx[0,0], intrinsic_mtx[1,1], intrinsic_mtx[0,2], intrinsic_mtx[1,2]

    x = np.zeros(depth.shape)
    y = np.zeros(depth.shape)
    z = depth / depth_scale
    for i in range(x.shape[1]):
        x[:,i] = i
    x = ((x - x.shape[1] / 2) * z) / fx
    for i in range(y.shape[0]):
        y[i,:] = i
    y = ((y - y.shape[0] / 2) * z) / fy
    
    x, y, z = x[:, 200:], y[:, 200:], z[:, 200:]
    x, y, z = x.reshape((-1,1)), y.reshape((-1,1)), z.reshape((-1,1))
    r, g, b = rgb[:,0].reshape((-1,1)), rgb[:,1].reshape((-1,1)), rgb[:,2].reshape((-1,1))
    valid = ((z < 0.002) & (z > 0)).reshape(-1)
    x, y, z = x[valid], y[valid], z[valid]
    r, g, b = r[valid], g[valid], b[valid]
    points = np.concatenate((x, -y, -z), axis=1)
    colors = np.concatenate((r, g, b), axis=1) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def local_icp_algorithm(pcd1, pcd2, trans_init, threshold):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
    # draw_registration_result(pcd1, pcd2, reg_p2p.transformation, paint_uniform_color=False)
    return reg_p2p.transformation


def transform_pose(pose, tx) :
    transform_ = sl.Transform()
    transform_.set_identity()
    # Translate the tracking frame by tx along the X axis
    transform_[0][3] = tx
    # Pose(new reference frame) = M.inverse() * pose (camera frame) * M, where M is the transform between the two frames
    transform_inv = sl.Transform()
    transform_inv.init_matrix(transform_)
    transform_inv.inverse()
    pose = transform_inv * pose * transform_


def local_icp_algorithm_own(pcd1, pcd2, trans_init, threshold):
    # pass
    max_iterations = 1000
    source = copy.deepcopy(pcd1)
    target = copy.deepcopy(pcd2)
    
    source = np.asarray(source.points)
    target = np.asarray(target.points)

    m = source.shape[1]

    source_temp = np.ones((m+1, source.shape[0]))
    target_temp = np.ones((m+1, target.shape[0]))
    
    source_temp[:m,:] = np.copy(source.T)
    target_temp[:m,:] = np.copy(target.T)
    
    source_temp = trans_init @ source_temp

    prev_error = 0
    
    for i in range(max_iterations):
        # find the nearest neighbours between the current source and destination points
        neigh = NearestNeighbors(n_neighbors=1, radius=threshold, algorithm='auto')
        neigh.fit(target_temp[:m,:].T)
        distances, indices = neigh.kneighbors(source_temp[:m,:].T)
        indices = indices.reshape(-1)
        distances = distances.reshape(-1)
        valid = distances < threshold
        source_temp = source_temp[:,valid]
        target_temp = target_temp[:,indices]
        target_temp = target_temp[:,valid]
        source = source[valid,:]
        # compute the transformation between the current source and nearest destination points
        T = best_fit_transform(source_temp[:m,:].T, target_temp[:m,:].T)

        # update the current source
        source_temp = T @ source_temp

        # check error
        mean_error = np.sum(distances) / distances.size
        # print(mean_error)
        if abs(prev_error - mean_error) < 0.0001:
            print(i)
            break
        prev_error = mean_error

    # calculcate final tranformation
    T = best_fit_transform(source, source_temp[:m,:].T)

    return T

def best_fit_transform(source, target):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert source.shape == target.shape

    # get number of dimensions
    m = source.shape[1]

    # translate points to their centroids
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)
    Source = source - centroid_source
    Target = target - centroid_target
    # print(Source.shape)

    # rotation matrix
    W = Target.T @ Source # mxN @ Nxm
    U, S, Vt = np.linalg.svd(W)
    R = U @ Vt
    # print(R.shape)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = U @ Vt

    # translation
    t = centroid_target.T - R @ centroid_source.T

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T


if __name__ == '__main__':
    args = parse_config()
    num_files = len(os.listdir(args.scene))
    intrinsic_mtx = np.array([[679.80212402, 0.0, 593.40826416],
                              [0.0, 679.80212402, 357.66592407],
                              [0.0, 0.0, 1.0]])
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        1280, 720, 679.80212402, 679.80212402, 593.40826416, 357.66592407
    )
    voxel_size = 0.00006
    threshold = voxel_size * 0.4
    trans_mtx_list = []
    pcd_list = []
    image_list = os.listdir(args.scene)
    image_list = sorted(image_list)
    if 'pose_data.npy' in image_list:
        image_list.remove('pose_data.npy')

    for i in range(0, num_files//2):
        print(os.path.join(args.scene, image_list[i].replace('depth', 'left')))
        rgb_path = os.path.join(args.scene, image_list[i].replace('depth', 'left'))
        depth_path = os.path.join(args.scene, image_list[i])
        pcd = depth_image_to_point_cloud(rgb_path, depth_path, intrinsic_mtx)
        pcd_list.append(pcd)
    num_pcd = len(pcd_list)
    
    if args.mode == 0:
        pose_data_list = np.load(os.path.join(args.scene, 'pose_data.npy'))
        pose_data_list[:,[0,1,2],3] /= 100000.0
        for i in range(len(pose_data_list)):
            pcd_list[i] = pcd_list[i].transform(pose_data_list[i])
            print(pose_data_list[i])
        pcd_total = o3d.geometry.PointCloud()
        for i in range(num_pcd):
            pcd_total += pcd_list[i]
        o3d.visualization.draw_geometries([pcd_total])

    elif args.mode == 1:
        for i in range(num_pcd-1):
            print(f'{num_pcd-i-1}->{num_pcd-i-2}:')
            source = pcd_list[num_pcd-i-1]
            target = pcd_list[num_pcd-i-2]
            source, target, source_down, target_down, source_fpfh, target_fpfh = \
                prepare_dataset(source, target, voxel_size)
            start = time.time()
            result_ransac = execute_global_registration(source_down, target_down,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size)  
            
            transformation = local_icp_algorithm(source_down, target_down, result_ransac.transformation, threshold)
            trans_mtx_list.append(transformation)
        trans_mtx_list.reverse()
        for i in range(1, len(trans_mtx_list)):
            trans_mtx_list[i] = trans_mtx_list[i-1] @ trans_mtx_list[i]
    
        for i in range(len(trans_mtx_list)):
            pcd_list[i+1].transform(trans_mtx_list[i])

        pcd_total = o3d.geometry.PointCloud()
        for i in range(num_pcd):
            pcd_total += pcd_list[i]
        o3d.visualization.draw_geometries([pcd_total])
    
    elif args.mode == 2:
        pose_data_list = np.load(os.path.join(args.scene, 'pose_data.npy'))
        pose_data_list[:,[0,1,2],3] /= 100000.0
        for i in range(len(pose_data_list)):
            pcd_list[i] = pcd_list[i].transform(pose_data_list[i])
        
        for i in range(num_pcd-1):
            print(f'{num_pcd-i-1}->{num_pcd-i-2}:')
            source = pcd_list[num_pcd-i-1]
            target = pcd_list[num_pcd-i-2]
            source, target, source_down, target_down, source_fpfh, target_fpfh = \
                prepare_dataset(source, target, voxel_size)
            start = time.time()
            result_ransac = execute_global_registration(source_down, target_down,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size)  
            
            transformation = local_icp_algorithm(source_down, target_down, np.identity(4), threshold)
            trans_mtx_list.append(transformation)
        trans_mtx_list.reverse()
        for i in range(1, len(trans_mtx_list)):
            trans_mtx_list[i] = trans_mtx_list[i-1] @ trans_mtx_list[i]
    
        for i in range(len(trans_mtx_list)):
            pcd_list[i+1].transform(trans_mtx_list[i])

        pcd_total = o3d.geometry.PointCloud()
        for i in range(num_pcd):
            pcd_total += pcd_list[i]
        o3d.visualization.draw_geometries([pcd_total])
