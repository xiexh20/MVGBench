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
import random
import json

import numpy as np

from PIL import Image
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, CameraInfo, fov2focal, focal2fov
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, Camera


def look_at(eye, target, up=(0, 1, 0)):
    """
    eye: camera position
    target: target position
    returns a world to camera transform matrix 4x4
    """
    # Convert to numpy arrays
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)

    # Compute forward, right and up vectors
    forward = eye - target # forward: target --> camera, opposite to opengl camera
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    # Create rotation and translation matrices
    rotation = np.eye(4)
    rotation[:3, :3] = np.vstack([right, up, forward]) # vstack: forward is the 3rd row

    translation = np.eye(4)
    translation[:3, 3] = -eye

    # Combine rotation and translation to form view matrix
    view_matrix = np.dot(rotation, translation)

    return view_matrix

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print(f"Found transforms_train.json file, assuming Blender data set! file path: {args.source_path}")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval) # load train and test file
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print(f"Loading Training Cameras with resolution scale {resolution_scale}")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print(f"Loading Test Cameras with resolution scale {resolution_scale}")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getConsistencyCameras(self, scale=1.0):
        "camera poses for computing consistency"
        train_cameras = self.getTrainCameras(scale)
        test_cameras = self.getTestCameras(scale)
        height, width = test_cameras[0].image_height, test_cameras[0].image_width
        rot = train_cameras[0].R
        w2c = np.transpose(rot)
        w2c[:3, 1:3] *= -1 # convert back to blender
        forward = w2c[2, :3]  # target -->camera, forward is the 3rd row
        x, y, z = forward
        elev = np.rad2deg(np.arcsin(z))
        # get 8 views above and 8 views bottom

        off = self.args.elev_offset
        azims = np.arange(8.5, 360, 360/8)
        elevs_rad = [np.deg2rad(min(elev+off, 89))] * len(azims) + [np.deg2rad(max(elev-off, -89))] * len(azims)
        azims = np.deg2rad(np.concatenate([azims, azims]))
        fov = 42
        dist = 3.2
        locations = [(dist * np.sin(azim_i) * np.cos(elev_rad), -dist * np.cos(elev_rad) * np.cos(azim_i),
                      np.sin(elev_rad) * dist) for azim_i, elev_rad in zip(azims, elevs_rad)]
        transform_all = [np.linalg.inv(look_at(loc, (0, 0, 0.), up=(0, 0, 1.))) for loc in locations]
        print(f"Elevation angle: {elev:.3f}, offset: {off:.3f}")

        cameras_info = []
        for idx, mat in enumerate(transform_all):
            c2w = mat
            c2w[:3, 1:3] *= -1 # blender to opencv

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            fovx = np.deg2rad(fov)
            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy
            FovX = fovx

            cameras_info.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                                           image=Image.fromarray(np.zeros((height, width, 3)).astype(np.uint8)),
                                        image_path='', image_name=f'{idx:03d}', width=width,
                                        height=height))
        cameras_consis = cameraList_from_camInfos(cameras_info, 1.0, self.args)
        return cameras_consis
