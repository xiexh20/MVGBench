#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import time

from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"] # rotation is not used in the model

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            # import pdb; pdb.set_trace()
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            if os.path.isfile(image_path):
                image = Image.open(image_path)
            else:
                print(f'Warning: {image_path} is not a file, using dummy data.')
                image = Image.fromarray(np.zeros((256,256,4), dtype=np.uint8), "RGBA")
            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # import pdb;pdb.set_trace()
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000

        # We create random points inside the bounds of the synthetic Blender scenes
        # bound = 2.6 # default NeRF
        bound = 2.0 # in our case, we don't need that large bound! since Nov12 1:22am.
        xyz = np.random.random((num_pts, 3)) * bound - bound/2. #
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        print(f"Generating random point cloud ({num_pts}) from bound {bound}...")
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

        # # Test: use carving to init points
        # start = time.time()
        # images = [info.image for info in train_cam_infos]
        # poses = []
        # for idx, info in enumerate(train_cam_infos):
        #     R, T = info.R, info.T
        #     w2c = np.eye(4)
        #     w2c[:3, :3] = np.transpose(R)
        #     w2c[:3, 3] = T
        #     c2w = np.linalg.inv(w2c)
        #     poses.append(c2w)
        # xyz = occ_from_sparse_initialize(np.stack(poses), images, train_cam_infos, 256, num_pts)
        # shs = np.ones((num_pts, 3)) * 0.2
        # end = time.time()
        # print(f"Time to init carving: {end-start:.4f}") # 10s in first two example, then ~3.5s for later examples
        # storePly(ply_path, xyz, SH2RGB(shs) * 255)
        # # End of test with carving


    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def occ_from_sparse_initialize(poses, images, cameras, grid_reso, num_points):
    """

    :param poses: Nx4x4, camera to world transform, which format?
    :param images: np array list, in range [0, 255]
    :param cameras: train_cam_infos,
    :param grid_reso: 256
    :param num_points: 100000, points to sample on the surface
    :return:
    """
    import rembg, torch, trimesh, mcubes
    from scene.cameras import Camera

    # fov is in degrees
    this_session = rembg.new_session()
    imgs = [rembg.remove(im, session=this_session) for im in images] # this returns rgba image

    reso = grid_reso
    occ_grid = torch.ones((reso, reso, reso), dtype=torch.bool, device="cuda")

    c2ws = poses
    # center = c2ws[..., :3, 3].mean(axis=0)
    # radius = np.linalg.norm(c2ws[..., :3, 3] - center, axis=-1).mean()
    # xx, yy, zz = torch.meshgrid(
    #     torch.linspace(-radius, radius, reso, device="cuda"),
    #     torch.linspace(-radius, radius, reso, device="cuda"),
    #     torch.linspace(-radius, radius, reso, device="cuda"),
    #     indexing="ij",
    # )

    # simply assume the camera poses are already normalized and centered
    center, radius = np.zeros((3,)), 1.0
    xx, yy, zz = torch.meshgrid(
        torch.linspace(-radius, radius, reso, device="cuda"),
        torch.linspace(-radius, radius, reso, device="cuda"),
        torch.linspace(-radius, radius, reso, device="cuda"),
        indexing="ij",
    )
    # print(f"camera center: {center}, radius {radius}")

    # xyz_grid = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=-1)
    ww = torch.ones((reso, reso, reso), dtype=torch.float32, device="cuda")
    xyzw_grid = torch.stack((xx, yy, zz, ww), dim=-1)
    xyzw_grid[..., :3] += torch.from_numpy(center).cuda() # why adding this???

    # c2ws = torch.tensor(c2ws, dtype=torch.float32)

    for c2w, camera, img in zip(c2ws, cameras, imgs):
        img = np.asarray(img)
        alpha = img[..., 3].astype(np.float32) / 255.0
        is_foreground = alpha > 0.05
        is_foreground = torch.from_numpy(is_foreground).cuda()

        full_proj_mtx = Camera(
            colmap_id=camera.uid,
            R=camera.R,
            T=camera.T,
            FoVx=camera.FovX,
            FoVy=camera.FovY,
            image=torch.randn(3, 10, 10),
            gt_alpha_mask=None,
            image_name="no",
            uid=0,
            data_device="cuda",
        ).full_proj_transform

        ij = xyzw_grid @ full_proj_mtx  # (reso, reso, reso, 4)
        ij[..., :2] = ij[..., :2] / ij[..., 2:3] # this is important to have correct projection!

        ij = (ij + 1) / 2.0 # this value is too large!
        h, w = img.shape[:2]
        ij = ij[..., :2] * torch.tensor([w, h], dtype=torch.float32, device="cuda")
        ij = (
            ij.clamp(
                min=torch.tensor([0.0, 0.0], device="cuda"),
                max=torch.tensor([w - 1, h - 1], dtype=torch.float32, device="cuda"),
            )
            .to(torch.long)
            .cuda()
        )
        # import pdb; pdb.set_trace()
        occ_grid = torch.logical_and(occ_grid, is_foreground[ij[..., 1], ij[..., 0]])

    # To mesh
    occ_grid = occ_grid.to(torch.float32).cpu().numpy()
    vertices, triangles = mcubes.marching_cubes(occ_grid, 0.5)

    # import pdb; pdb.set_trace()
    vertices = vertices / (grid_reso - 1) * 2 - 1
    vertices = vertices * radius + center

    xyz = trimesh.Trimesh(vertices, triangles).sample(num_points)

    return xyz

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}