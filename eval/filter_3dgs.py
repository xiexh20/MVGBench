"""
filter existing 3dgs to keep gs only inside bound -1,1
"""
import sys, os
from datetime import datetime

import imageio, json
import pickle as pkl

import trimesh

sys.path.append(os.getcwd())
import numpy as np
import os.path as osp
import cv2
from glob import glob
from tqdm import tqdm
from plyfile import PlyData, PlyElement


def main(args):
    ""
    bound = 1.0
    # pc_files = sorted(glob(osp.join(args.folder, '*/point_cloud/iteration*/point_cloud.ply')))
    pc_files = ['/home/ubuntu/Documents/Avengers_Thor_PLlrpYniaeB_sv3d-filter.ply']
    for file in tqdm(pc_files):
        plydata = PlyData.read(file)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        max_sh_degree = 3 
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # now filter 
        m1 = (xyz[:, 0] <= 1.0) & (xyz[:, 1] <= 1.) & (xyz[:, 2] <= 1.) & (xyz[:, 0] >= -1.0) & (xyz[:, 1] >= -1.0) & (xyz[:, 2] >= -1.0)
        # opacity_real = 1. / (1 + np.exp(-opacities))
        # m1 = opacity_real[:, 0] < 0.1

        # do random sampling using the covariance matrix
        import pdb; pdb.set_trace()
        # for the white dots, the avg scale is large: 0.0134

        xyz = xyz[m1]
        opacities = opacities[m1]
        f_dc = features_dc[m1].reshape((xyz.shape[0], -1))
        f_rest = features_extra[m1].reshape((xyz.shape[0], -1))
        scales = scales[m1]
        rots = rots[m1]

        attribut_list = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
                         'f_rest_0', 'f_rest_1', 'f_rest_2', 'f_rest_3', 'f_rest_4', 'f_rest_5', 'f_rest_6',
                         'f_rest_7', 'f_rest_8', 'f_rest_9', 'f_rest_10', 'f_rest_11', 'f_rest_12', 'f_rest_13',
                         'f_rest_14', 'f_rest_15', 'f_rest_16', 'f_rest_17', 'f_rest_18', 'f_rest_19', 'f_rest_20',
                         'f_rest_21', 'f_rest_22', 'f_rest_23', 'f_rest_24', 'f_rest_25', 'f_rest_26', 'f_rest_27',
                         'f_rest_28', 'f_rest_29', 'f_rest_30', 'f_rest_31', 'f_rest_32', 'f_rest_33', 'f_rest_34',
                         'f_rest_35', 'f_rest_36', 'f_rest_37', 'f_rest_38', 'f_rest_39', 'f_rest_40', 'f_rest_41',
                         'f_rest_42', 'f_rest_43', 'f_rest_44', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0',
                         'rot_1', 'rot_2', 'rot_3']
        
        # now save
        dtype_full = [(attribute, 'f4') for attribute in attribut_list]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        normals = np.zeros_like(xyz)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        # outfile = file.replace('.ply', '_filtered.ply')
        # outfile = file
        outfile = file.replace('.ply', '_small-opacity.ply')
        PlyData([el]).write(outfile)
        # print('saved to', outfile)
        # break

def check_outliers():
    "check how many points are outliers"
    pc_files = sorted(glob('output/consistency/sv3dp+sv3d-v21-elev-*/*/point_cloud/iteration*/point_cloud.ply'))
    count_out, count_total = 0, 0
    percentage = []
    for file in tqdm(pc_files):
        if 'bbox2.6' in file:
            continue
        pc = trimesh.load(file)
        xyz = np.array(pc.vertices)
        m1 = (xyz[:, 0] <= 1.05) & (xyz[:, 1] <= 1.05) & (xyz[:, 2] <= 1.05) & (xyz[:, 0] >= -1.05) & (xyz[:, 1] >= -1.05) & (xyz[:, 2] >= -1.05)
        count_total += len(m1) # does filtering affect the CD error a lot? without filter, the CD error can be even smaller
        count_out += np.sum(~m1)
        percentage.append(np.sum(~m1)/len(m1))
    print(f"In total {len(percentage)} examples, outlier: {count_out/count_total:.3f}, avg: {np.mean(percentage):.3f}")
    # In total 30 examples, outlier: 0.056, avg: 0.059
    # all elevation angles: In total 180 examples, outlier: 0.106, avg: 0.145
    # both even and odd: In total 360 examples, outlier: 0.106, avg: 0.146
    # with bbox size 1.1: In total 360 examples, outlier: 0.044, avg: 0.077
    # with bbox size 1.05: In total 360 examples, outlier: 0.068, avg: 0.106


def test_filter_gs():
    "test the property of filtered 3dgs"
    from gaussian_renderer import GaussianModel

    file = f'/home/ubuntu/Documents/Avengers_Thor_PLlrpYniaeB_sv3d-filter-3pts.ply'
    file_full = 'output/consistency/sv3dp-lvis-static+gso100-sv3d-v21-elev012-amb1.0+step-010500-merged_even/Avengers_Thor_PLlrpYniaeB/point_cloud/iteration_10000/point_cloud.ply'
    gaussian = GaussianModel(sh_degree=3)
    gaussian.load_ply(file)
    cov = gaussian.get_covariance(strip=False).detach().cpu().numpy() # (N. 3. 3), this seems to be in world space.
    scales = gaussian.get_scaling
    opacity = gaussian.get_opacity
    np.set_printoptions(suppress=True, precision=8)

    # do some random sample and save
    centers = gaussian.get_xyz.detach().cpu().numpy()
    N = len(centers)
    # Perform Cholesky decomposition for each covariance matrix
    chol_decompositions = np.linalg.cholesky(cov)  # N x 3 x 3, need transpose to align the direction!
    print('covariance:', cov)
    print('decompositions:', chol_decompositions)
    # eigvals = np.linalg.eigvals(cov)
    # print('eigen values:', eigvals)
    samples_all = []
    num_points = 200
    for _ in range(num_points):
        samples = np.random.randn(N, 3)
        samples = centers + np.einsum('ijk,ik->ij', chol_decompositions, samples)
        # trimesh.PointCloud(samples).export(file.replace('.ply', f'_sample-T{_:02d}.ply'))
        samples_all.append(samples)
    samples_all = np.stack(samples_all, axis=0) # N, 3, 3

    # Step 1: Eigen decomposition of covariance matrices
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # Shape: (N, 3), (N, 3, 3)
    semi_axes = np.sqrt(eigenvalues)  # Semi-axis lengths, shape: (N, 3)
    # for _ in range(1000):
    # Step 2: Sample points uniformly on the unit sphere
    total_points = N * num_points
    phi = np.arccos(2 * np.random.rand(total_points) - 1)  # Shape: (N * num_points,)
    theta = 2 * np.pi * np.random.rand(total_points)  # Shape: (N * num_points,)

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    unit_sphere_points = np.stack((x, y, z), axis=-1)  # Shape: (N * num_points, 3)

    # Step 3: Scale by semi-axes
    unit_sphere_points = unit_sphere_points.reshape(N, num_points, 3)  # Shape: (N, num_points, 3)
    scaled_points = unit_sphere_points * semi_axes[:, np.newaxis, :]  # Shape: (N, num_points, 3)

    # Step 4: Rotate by eigenvectors
    rotated_points = np.einsum('nij,nmj->nmi', eigenvectors, scaled_points)  # Shape: (N, num_points, 3)

    # Step 5: Translate by centers
    translated_points = rotated_points + centers[:, np.newaxis, :]  # Shape: (N, num_points, 3)

    for i in range(N):
        trimesh.PointCloud(translated_points[i]).export(file.replace('.ply', f'_sample-1std-batch{i:02d}.ply'))
        trimesh.PointCloud(samples_all[:, i]).export(file.replace('.ply', f'_sample-full-cov-{i:02d}.ply'))

    # sample points from 1std
    # for i in range(N):
    #     covariance = cov[i]
    #     num_points = 1000
    #     # Eigenvalue decomposition of the covariance matrix
    #     eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    #
    #     # Semi-axis lengths
    #     semi_axes = np.sqrt(eigenvalues)
    #
    #     # Sample points on a unit sphere using spherical coordinates
    #     phi = np.arccos(1 - 2 * np.random.rand(num_points))  # Polar angle
    #     theta = 2 * np.pi * np.random.rand(num_points)  # Azimuthal angle
    #
    #     # Parametrize the unit sphere
    #     x = np.cos(theta) * np.sin(phi)
    #     y = np.sin(theta) * np.sin(phi)
    #     z = np.cos(phi)
    #     sphere_points = np.stack((x, y, z), axis=-1)  # Shape (num_points, 3)
    #
    #     # Scale by semi-axis lengths
    #     scaled_points = sphere_points * semi_axes  # Element-wise multiplication
    #
    #     # Rotate by eigenvectors and shift by center
    #     ellipsoid_points = scaled_points @ eigenvectors.T + centers[i]
    #     trimesh.PointCloud(ellipsoid_points).export(file.replace('.ply', f'_sample-1std{i:02d}.ply'))
    #     trimesh.PointCloud(samples_all[:, i]).export(file.replace('.ply', f'_sample-full-cov-{i:02d}.ply'))

    # gfull = GaussianModel(sh_degree=3)
    # gfull.load_ply(file_full)
    # cov_full = gfull.get_covariance(strip=False)
    # scales_full = gfull.get_scaling
    # opacity_full = gfull.get_opacity # statiscally these two gaussians are very similar


    # plydata = PlyData.read(file)

    # xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
    #                 np.asarray(plydata.elements[0]["y"]),
    #                 np.asarray(plydata.elements[0]["z"])), axis=1)
    # opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    # opacity = 1. / (1 + np.exp(-opacities)) # usually very large > 0.8
    # import pdb; pdb.set_trace()




if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('folder')

    args = parser.parse_args()

    # main(args)
    # check_outliers()
    test_filter_gs()