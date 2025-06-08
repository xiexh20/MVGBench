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
import json

import numpy as np
import torch
import trimesh, cv2

from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
# from gaussian_renderer import render
from gaussian_renderer import render as render
import torchvision
import os.path as osp
from glob import glob
from eval.metric_utils import normalize_percentile
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, quiet=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")

    makedirs(render_path, exist_ok=True)
    if name == 'train':
        makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    loop = tqdm(views, desc="Rendering progress") if not quiet else views

    # need to scale covariance
    # ratio = 1.5
    # gaussians._xyz *= ratio
    # gaussians._scaling = gaussians._scaling + np.log(ratio) # multiply becomes addition in exponential space

    for idx, view in enumerate(loop):
        # view.T *= ratio # multiply camera as well, this somehow is not working
        output = render(view, gaussians, pipeline, background)
        rendering = output["render"]
        # alpha = output["alpha"]
        # rendering = torch.cat([rendering, alpha], dim=0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        dmap = (output['depth'].cpu().numpy() * 65535 / 10.).astype(np.uint16)[0]
        if name != 'train':
            # dmap = (dmap /dmap.max() * 255 ).astype(np.uint8)
            cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), dmap)
        if name == 'train':
            # only save train-gt
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png")) # do not save gt

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, quiet=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree) # empty scene
        # TODO: change cameras here to render random views
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False) # TODO: why here the train cameras also include test?? -> need to specify --eval otherwise all images are used in training
        # import pdb;pdb.set_trace()
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if dataset.normalize_gs: # normalize the 3dgs for real image testing
            args.test_name = 'align-icp'
            # try to load from disk the normalize params
            part = osp.basename(osp.dirname(dataset.model_path)).split("_")[-1] # either even or odd
            other = 'odd' if 'even' in part else 'even'
            norm_file = osp.join(dataset.model_path, 'transform_icp.json')
            if not osp.isfile(norm_file):
                norm_file = osp.join(dataset.model_path.replace('_' + part, '_' + other), 'transform_icp.json')
            if not osp.isfile(norm_file):
                raise ValueError(f"cannot find {norm_file}!")
                # compute params online
                pts1 = gaussians.get_xyz.cpu().numpy()
                pts2 = np.array(trimesh.load(osp.join(dataset.model_path.replace('_' + part, '_' + other),
                                                      f'point_cloud/iteration_{iteration}/point_cloud.ply'),
                                             process=False).vertices)
                vcen, obj_size = normalize_percentile(np.concatenate([pts1, pts2], 0))
                # save results
                norm_file = osp.join(dataset.model_path, 'transform_icp.json')
                json.dump({"cent": vcen.tolist(), 'scale': obj_size, }, open(norm_file, 'w'))
                json.dump({"cent": vcen.tolist(), 'scale': obj_size, }, open(norm_file.replace('_' + part, '_' + other), 'w'))
            else:
                # load from file
                d = json.load(open(norm_file, 'r'))
                transforms = np.array(d['transformation']) # 4x4 matrix
                # scale the centers, consider only scale and translation
                u, s, vt = np.linalg.svd(transforms[:3, :3]) # get the scale out
                scale = torch.from_numpy(s).float().to(gaussians._xyz.device)
                center = torch.from_numpy(transforms[:3, 3]).float().to(gaussians._xyz.device)
                gaussians._xyz = gaussians._xyz * scale[None] + center[None]

                # Full transformation with rotation as well -> does not change much
                # mat = torch.from_numpy(transforms).float().to(gaussians._xyz.device)
                # gaussians._xyz = torch.matmul(gaussians._xyz, mat[:3, :3].T) + mat[:3, 3][None]
                gaussians._scaling = gaussians._scaling + torch.log(scale[None].to(gaussians._scaling.device)) # the covariance scale should be updated as well!
                # import pdb;pdb.set_trace()

                # vcen, obj_size = np.array(d['cent']), d['scale']
            # run normalization, TODO: the scale should be applied to the covariance matrix as well!
            # xyz = (gaussians._xyz - torch.from_numpy(vcen[None]).float().to(gaussians._xyz.device))/obj_size
            # gaussians._xyz = xyz
            # args.test_name = 'norm' # save to another path
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, quiet)

        if not skip_test:
             render_set(dataset.model_path, args.test_name, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, quiet)
        if dataset.elev_offset >= 0.:
            render_set(dataset.model_path, f'elev+-{dataset.elev_offset}', scene.loaded_iter, scene.getConsistencyCameras(), gaussians, pipeline, background, quiet)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=10000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--test_name', default='test')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)

    args = parser.parse_args()

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render multiple folders sequentially
    if osp.isdir(args.model_path):
        folders = [args.model_path]
    else:
        folders = sorted(glob(args.model_path))

    end = len(folders) if args.end is None else args.end
    loop = tqdm(folders[args.start:end])
    for folder in loop:
        ss = folder.split('/')
        name = osp.basename(folder)  # scan name
        out_dir = f'output/consistency/{ss[-2]}/{name}'
        if '/old/' in folder:
            out_dir = f'output/consistency/old/{ss[-2]}/{name}'
        args.model_path = out_dir
        args.quiet = True if len(folders) > 1 else False
        # args.source_path = out_dir, don't do this, the src parth will be loaded from cfg file automatically

        # now merge args from config file
        args_run = get_combined_args(args)
        if 'normalize_gs' not in args_run:
            args_run.normalize_gs = False
        try:
            render_sets(model.extract(args_run), args_run.iteration, pipeline.extract(args_run), args_run.skip_train,
                        args_run.skip_test, args.quiet)
        except Exception as e:
            import traceback
            print(traceback.format_exc())

    # args = get_combined_args(parser)
    # print("Rendering " + args.model_path)

    # for it in args.iteration:
    #     render_sets(model.extract(args), it, pipeline.extract(args), args.skip_train, args.skip_test)

