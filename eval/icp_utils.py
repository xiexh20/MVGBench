import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation


def reg_icp(source: np.ndarray, target: np.ndarray, voxel_size, init=np.eye(4), method='point2point', scaling=False):
    """
    iterative icp in 3 steps
    """
    assert method in ['point2plane', 'point2point']
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane() if method == 'point2plane' else o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # estimation.with_scaling = scaling
    voxel_radius = [voxel_size*8, voxel_size*4, voxel_size]
    max_iter = [30, 20, 100]
    # max_iter = [50, 30, 20]
    voxel_radius = [voxel_size * 8, voxel_size * 4, voxel_size*1.5]
    max_iter = [30, 20, 50]
    current_trans = init.copy()
    for it, radius in zip(max_iter, voxel_radius):
        target_down = target.voxel_down_sample(voxel_size=radius)
        source_down = source.voxel_down_sample(voxel_size=radius)
        if method == 'point2plane':
            target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        if radius <= 2*voxel_size:
            estimation.with_scaling = scaling # activate scale only in the last round
        # increase correspondence threshold can improve!
        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, radius*4, current_trans,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=it)) # segmentation fault!
        # check the scale after each round, the scale is uniform across 3 axis, but close to 1, e.g. 0.96-0.98
        # u, s, vt = np.linalg.svd(np.array(result_icp.transformation)[:3, :3])
        # print(f'iteration {it} fitness: {result_icp.fitness:.2f}, scales: {s}, {estimation.with_scaling}')
        if result_icp.fitness > 0.1:
            current_trans = result_icp.transformation

    return result_icp