"""
evaluate the 3D consistency of SV3D images via rendering and 3DGS centers
"""
import sys, os
from datetime import datetime

import json
import pickle as pkl
sys.path.append(os.getcwd())
import numpy as np
import os.path as osp
import cv2
from tqdm import tqdm
import trimesh, torch
import torch.nn.functional as F
from glob import glob
from eval.chamfer_distance import chamfer_distance, compute_fscore
from skimage.metrics import structural_similarity as calculate_ssim
from plyfile import PlyData, PlyElement
from skimage.metrics import peak_signal_noise_ratio
from eval.cleanfid import fid as FID

import lpips
LPIPS = lpips.LPIPS(net='alex', version='0.1')


def calc_2D_metrics(pred_np, gt_np, mask=None):
    # pred_np: [H, W, 3], [0, 255], np.uint8
    pred_image = torch.from_numpy(pred_np).unsqueeze(0).permute(0, 3, 1, 2)
    gt_image = torch.from_numpy(gt_np).unsqueeze(0).permute(0, 3, 1, 2)
    # [0-255] -> [-1, 1]
    pred_image = pred_image.float() / 127.5 - 1
    gt_image = gt_image.float() / 127.5 - 1
    # for 1 image
    # pixel loss
    loss = F.mse_loss(pred_image[0], gt_image[0].cpu()).item()
    # LPIPS
    lpips = LPIPS(pred_image[0], gt_image[0].cpu()).item()  # [-1, 1] torch tensor
    # SSIM
    ssim = calculate_ssim(pred_np, gt_np, channel_axis=2)
    # PSNR
    if mask is not None:
        gt, pr = gt_np[mask], pred_np[mask]
        psnr = peak_signal_noise_ratio(gt, pr)
    else:
        psnr = cv2.PSNR(gt_np, pred_np)

    return loss, lpips, ssim, psnr



class BaseEvaluator:
    def __init__(self):
        ""
        self.m2cm = 100 # multiplier for unit conversion
        self.outdir = 'results'
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(osp.join(self.outdir, 'raw'), exist_ok=True)

    def load_filtered_gaussians(self, file):
        plydata1 = PlyData.read(file)
        pts1 = np.stack((np.asarray(plydata1.elements[0]["x"]),
                         np.asarray(plydata1.elements[0]["y"]),
                         np.asarray(plydata1.elements[0]["z"])), axis=1)
        opacity1 = np.asarray(plydata1.elements[0]["opacity"])
        opacity = 1. / (1 + np.exp(-opacity1))
        pts1 = pts1[opacity >= 0.01]
        return pts1

    @staticmethod
    def get_parser():
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('-ne', '--name_even')
        parser.add_argument('-no', '--name_odd')
        parser.add_argument('-i', '--identifier')
        parser.add_argument('-reso', type=int, default=256)
        parser.add_argument('-filter', default=False, action='store_true')
        parser.add_argument('-it', '--iteration', default=10000, type=int)
        parser.add_argument('-bf', '--box_filter', default=False, action='store_true')
        parser.add_argument('-p', '--percentile', type=float, default=10)
        parser.add_argument('-tn', '--test_name', default='test')
        parser.add_argument('-np', '--normalize_percentile', default=False, action='store_true')
        parser.add_argument('-d', '--dataset_path', help='used to compute FID', default='/home/ubuntu/efs/static/GSO30')
        parser.add_argument('-rn', '--render_name', help='used to computed FID', default='zero123-v21-nolight')
        parser.add_argument('--nofid', default=False, action='store_true')
        parser.add_argument('--n_eval', default=None, type=int, help='evaluate how many examples')

        parser.add_argument('--random_order', default=False, action='store_true') # randomize the order of objects
        return parser

    def get_metric_keys(self):
        "the name for the metrics"
        return self.get_metrics_2d() + self.get_metrics_3d() # + ['FID']

    def get_metrics_2d(self):
        return ["cPSNR", 'cSSIM', 'cLPIPS']

    def get_metrics_3d(self):
        return ["Chamf", 'F-score', 'Hausdorff']

    def get_gt_files(self, dataset_path, name_even, num_eval=None, order_indices=None):
        "get a list of gt image files to compute fid"
        name_even = osp.basename(name_even)
        if '+' not in name_even:
            render_name = name_even.split('_')[0]#[:-4]
        else:
            render_name = name_even.split('_')[0].split('+')[1]
        if 'gso100' in render_name or 'sv3d-reso576' in render_name:
            dataset_path = '/home/ubuntu/efs/static/gso100'
            n = 30
        elif 'omni202' in render_name:
            dataset_path = '/home/ubuntu/efs/static/omni202'
            n = 30
        elif 'mvpnet50' in render_name:
            dataset_path = '/home/ubuntu/efs/static/mvpnet50'
            n=1
        elif 'omni3d' in render_name:
            dataset_path = '/home/ubuntu/efs/static/OminiObj30'
            n = 30
        elif 'co3d2seq' in render_name:
            dataset_path = '/home/ubuntu/efs/static/co3dv2-manual'
            render_name = 'fid-16images'
            n = 16
        elif 'mvimgnet' in render_name:
            dataset_path = '/home/ubuntu/efs/static/mvimgnet230'
            render_name = 'fid-16images'
            n = 16
        elif 'views84' in render_name:
            dataset_path = '/home/ubuntu/efs/static/objaverse/static-64'
            n = 84
        elif 'train-as-test' in render_name:
            dataset_path = '/home/ubuntu/efs/static/objav100'
            n = 21
        elif 'objaveval' in render_name:
            dataset_path = '/home/ubuntu/efs/static/objav-eval'
            n = 16
        else:
            dataset_path = '/home/ubuntu/efs/static/GSO30'
            n=30
        # if 'GSO30' in dataset_path:
        #     # gso30
        #     render_names = ['imagedream-v21-nolight', 'mvdfusion-v21-nolight', 'sv3d-v21-nolight', 'zero123-v21-nolight']
        #     n = 30
        # elif 'gso100' in dataset_path:
        #     render_names = ['gso100-zero123-v21-elev000-amb1.0', 'gso100-imagedream-v21-elev000-amb1.0',
        #                     'gso100-mvdfusion-v16-elev030-amb1.0', 'gso100-sv3d-v21-elev012-amb1.0']
        #     n = 30
        # elif 'omni202' in dataset_path:
        #     render_names = ['omni202-imagedream-v21-elev002-amb1.0', 'omni202-mvdfusion-v16-elev030-amb1.0',
        #                     'omni202-sv3d-v21-elev012-amb1.0', 'omni202-zero123-v21-elev000-amb1.0']
        #     n = 30
        # elif 'mvpnet50' in dataset_path:
        #     render_names = ['mvpnet50-mvdfusion-v16-manual', 'mvpnet50-mvdfusion-v16-manual-elev030']
        #     n = 1 # all other frames are the same
        # elif 'co3dv2' in dataset_path:
        #     render_names = ['imagedream-v21-frame00001-elev030']
        #     n = 1  # all other frames are the same
        # else:
        #     raise ValueError(f"Unknown dataset {dataset_path}")
        print(f"GT files from {dataset_path}, {render_name}")
        render_names = [render_name]
        gt_files = []
        folders = sorted(glob(dataset_path+"/*"))
        if order_indices is not None:
            folders = [folders[x] for x in order_indices] # randomize the order
        end = len(folders) if num_eval is None else num_eval
        for folder in folders[:end]:
            for rn in render_names:
                files = sorted(glob(folder+"/"+rn+"/*.png")) + sorted(glob(folder+"/"+rn+"/*.webp"))
                gt_files.extend(files[:n])
        assert len(gt_files) > 0, 'no gt files found!'
        return gt_files

    def evaluate(self, args):
        ""
        name_even = args.name_even
        name_odd = args.name_odd

        if '*' in name_even:
            name_even = sorted(glob(name_even))
            if len(name_even) == 0:
                print("No path found in", name_even)
                return
            name_even = name_even[0]
        if "*" in name_odd:
            name_odd = sorted(glob(name_odd)) # allow evaluation directly after test
            if len(name_odd) == 0:
                print("No path found in", name_odd)
                return
            name_odd = name_odd[0]
        scan_names = sorted(os.listdir(name_even))
        metric_keys = self.get_metric_keys()

        errors_all = {k:[] for k in metric_keys}
        files_3d, files_rgb = [], []
        pc_count = []
        iteration, test_name = args.iteration, args.test_name
        images_mv, images_gt = [], [] # all mv images, network direct prediction, and gt rendering

        order_indices = np.arange(len(scan_names)) if not args.random_order else np.random.choice(len(scan_names), len(scan_names), replace=False)
        # images_gt = self.get_gt_files(args.dataset_path, name_even, args.n_eval, order_indices) # for computing FID
        scan_names = [scan_names[x] for x in order_indices]

        end = len(scan_names) if args.n_eval is None else args.n_eval
        for name in tqdm(scan_names[:end]):
            if name == 'teapot' and ('amb' in name_even and 'amb1.0' not in name_even):
                print(f'skipping teapot for {name_odd}') # for robustness w.r.t. lighting, skip this object as it is pure white
                continue
            fodd, feven = f'{name_odd}/{name}', f'{name_even}/{name}'
            # collect image files for fid
            mv_files = sorted(glob(feven + f'/train/ours_{iteration}/gt/*.png')) + sorted(glob(fodd + f'/train/ours_{iteration}/gt/*.png'))
            images_mv.extend(mv_files)
            # evaluate 3D
            complete = self.check_3d_files(feven, fodd, iteration, name, name_odd)
            if not complete:
                continue
            errors_3d = self.evaluate_3d(args, feven, fodd, iteration, pc_count)
            if len(errors_3d) == 0:
                print(f'Geometric consistency evaluation failed on {name_even}')
                continue
            for k, v in errors_3d.items():
                errors_all[k].append(v)

            # evaluate 2D texture consistency
            errors_2d_i = self.evaluate_2d(args, feven, fodd)
            if len(errors_2d_i) == 0:
                print(f'Texture consistency evaluation failed on {name_even}')
                continue

            for k, v in errors_2d_i.items():
                errors_all[k].append(v)

            # metadata
            files_3d.append((fodd, feven))

        # compute fid
        # if args.nofid:
        #     score = float('nan')
        # else:
        #     score = self.compute_fid(images_gt, images_mv)
        # errors_all['FID'] = score

        self.format_output(args, errors_all, files_3d)

    def evaluate_2d(self, args, feven, fodd):
        iteration, test_name = args.iteration, args.test_name
        files_odd = sorted(glob(osp.join(fodd, f'{test_name}/ours_{iteration}/renders/*.png')))
        files_even = sorted(glob(osp.join(feven, f'{test_name}/ours_{iteration}/renders/*.png')))
        if len(files_odd) != len(files_even):
            print(f'unequal number of files for {feven}({len(files_even)} even) vs. {fodd}({len(files_odd)} odd)!')
            return {}
        if len(files_odd) == 0:
            print(f"No images found in {feven}!")
            return {}
        reso = args.reso
        keys_2d = self.get_metrics_2d()
        errors_2d_i = {k: [] for k in keys_2d}
        for odd, even in zip(files_odd, files_even):
            o = cv2.resize(cv2.imread(odd)[:, :, ::-1].copy(), (reso, reso))
            e = cv2.resize(cv2.imread(even)[:, :, ::-1].copy(), (reso, reso))
            pix, lpip, ssim, psnr = calc_2D_metrics(o, e)
            for k, v in zip(keys_2d, [psnr, ssim, lpip]):
                errors_2d_i[k].append(v)
        return errors_2d_i

    def check_3d_files(self, feven, fodd, iteration, name, name_odd):
        pcfile_even = osp.join(feven, f'point_cloud/iteration_{iteration}/point_cloud.ply')
        pcfile_odd = osp.join(fodd, f'point_cloud/iteration_{iteration}/point_cloud.ply')
        if not osp.isfile(pcfile_odd) or not osp.isfile(pcfile_even):
            print(f'skipped {name}?{osp.isfile(pcfile_even)} or {name_odd}?{osp.isfile(pcfile_odd)} ')
            # continue
            return False # incomplete
        return True

    def compute_fid(self, images_gt, images_mv):
        """
        given two lists of image files, compute fid score
        :param images_gt:
        :param images_mv:
        :return:
        """
        # def fill_in_white_bkg(img_np):
        #     "fill in background color to white"
        #     mask = np.mean(img_np, axis=-1) <=0.00001
        #     img_np[mask] = 255 # background as white, with this: fid 125.30 -> 165.17
        #     # print(mask.shape, img_np.shape)
        #     Image.fromarray(img_np).save('data/test.png') # the image has lots of white background!
        #     return img_np
        # if use true white background: FID=89.80(0.00), gso100: FID: 126.76(elev30),62.89(elev12)
        # omni202: 40.29(elev0)
        # feat_model = FID.build_feature_extractor('clean', 'cuda', use_dataparallel=True)
        # np_feats_mv = FID.get_files_features(images_mv, feat_model)
        # np_feats_gt = FID.get_files_features(images_gt, feat_model)
        # TODO: implement work around to handle gt image with rgba
        # use clip, we need to use same GT image set!
        # gso100: 8.77 (sv3d-elev012), omni202: 4.76 (zero123-elev0), gso30: 9.66
        # syncdreamer: gso30: 13.34 (zero123-nolight), gso100: 7.42 (mvdfusion-elev30), omni202: 5.55 (mvdfusion-elev030)
        # syncdreamer: gso30: 9.66 (mvdfusion-nolight), gso100: 9.35 (sv3d-elev12), omni202: 7.41 (zero123-elev0)
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
        clip_fx = CLIP_fx("ViT-B/32", device="cuda") # Free3d setup
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip
        np_feats_mv = FID.get_files_features(images_mv, feat_model, custom_fn_resize=custom_fn_resize)
        np_feats_gt = FID.get_files_features(images_gt, feat_model, custom_fn_resize=custom_fn_resize)

        mu_mv = np.mean(np_feats_mv, axis=0)
        sigma_mv = np.cov(np_feats_mv, rowvar=False)
        mu_gt = np.mean(np_feats_gt, axis=0)
        sigma_gt = np.cov(np_feats_gt, rowvar=False)
        score = FID.frechet_distance(mu_mv, sigma_mv, mu_gt, sigma_gt) #
        return score

    def evaluate_3d(self, args, feven, fodd, iteration, pc_count):
        "compute 3d metrics for the two gaussians"
        pcfile_even = osp.join(feven, f'point_cloud/iteration_{iteration}/point_cloud.ply')
        pcfile_odd = osp.join(fodd, f'point_cloud/iteration_{iteration}/point_cloud.ply')
        # feven = osp.dirname(osp.dirname(osp.dirname(pcfile_even)))
        if args.filter:
            veven = self.load_filtered_gaussians(pcfile_even)
            vodd = self.load_filtered_gaussians(pcfile_odd)  # TODO: implement sampling using covariance
            print('loading from filtered Gaussians')
        else:
            pc_odd = trimesh.load(pcfile_odd)
            pc_even = trimesh.load(pcfile_even)  # the range is roughly [-1, 1]

            # filter points outside the cube
            vodd = np.array(pc_odd.vertices)
            veven = np.array(pc_even.vertices)
            # filter with bbox: only keep points inside bbox
            if args.box_filter:
                m1 = (vodd[:, 0] <= 1.0) & (vodd[:, 1] <= 1.) & (vodd[:, 2] <= 1.) & (vodd[:, 0] >= -1.0) & (
                        vodd[:, 1] >= -1.0) & (vodd[:, 2] >= -1.0)
                vodd = vodd[m1]
                m2 = (veven[:, 0] <= 1.0) & (veven[:, 1] <= 1.) & (veven[:, 2] <= 1.) & (veven[:, 0] >= -1.0) & (
                        veven[:, 1] >= -1.0) & (veven[:, 2] >= -1.0)
                veven = veven[m2]

            # normalize points to a bbox

            # vmin = verts.min(axis=0)
            # vmax = verts.max(axis=0)
            # vcen = (vmin + vmax) / 2
            # obj_size = np.abs(verts - vcen).max()
            # use robust scale and center

            # q1 = np.percentile(verts, 25, axis=0)
            # q3 = np.percentile(verts, 75, axis=0)
            # obj_size = 1.5 * np.median(q3 - q1)
            # q1 = np.percentile(verts, 10, axis=0) # robust2
            # q3 = np.percentile(verts, 90, axis=0)
            # obj_size = np.median(q3 - q1)

            # adapt the scale based on noise level
            # sample first
            # N = 60000
            # indices = np.random.choice(len(vodd), N, replace=len(vodd) < N)
            # vodd = np.array(vodd)[indices]
            # indices = np.random.choice(len(veven), N, replace=len(veven) < N)
            # veven = np.array(veven)[indices]

            # nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(verts)
            # distances, _ = nbrs.kneighbors(verts)
            # local_densities = 1 / (np.mean(distances, axis=1) + 1e-6)  # prevent zero
            # density_std = np.std(local_densities)
            # normalized_noise = density_std / np.mean(local_densities)
            # normalized_noise = np.clip(normalized_noise, 1e-6, 1.) # need clip, otherwise it goes beyond
            # q = normalized_noise * 10

            N = 60000
            indices = np.random.choice(len(vodd), N, replace=len(vodd) < N)
            vodd = np.array(vodd)[indices]
            indices = np.random.choice(len(veven), N, replace=len(veven) < N)
            veven = np.array(veven)[indices]

            if args.normalize_percentile:
                # vcen, obj_size = normalize_percentile(np.concatenate((vodd, veven), axis=0), args.percentile)
                # # vcen, obj_size, _ = normalize_adaptive(np.concatenate((vodd, veven), axis=0), args.percentile)
                # vodd = (vodd - vcen) / obj_size
                # veven = (veven - vcen) / obj_size
                # load from file of ICP results
                nfile = f'{feven}/transform_icp.json'
                mat = np.array(json.load(open(nfile, 'r'))['transformation'])
                veven = np.matmul(veven, mat[:3, :3].T) + mat[:3, 3]
                vodd = np.matmul(vodd, mat[:3, :3].T) + mat[:3, 3]
        # fscore, cd = compute_fscore(pts_odd.copy(), pts_even.copy(), thres=0.02)
        # do FPS
        # pts_odd = np.array(pc_odd.vertices)
        # if len(pts_odd) < N:
        #     pts_odd = pts_odd[np.random.choice(len(pts_odd), N, replace=True)]
        # pts_even = np.array(pc_even.vertices)
        # if len(pts_even) < N:
        #     pts_even = pts_even[np.random.choice(len(pts_even), N, replace=True)]
        # pts_odd = sample_farthest_points(torch.from_numpy(pts_odd).float()[None], torch.Tensor([N]))[0][0].numpy()
        # pts_even = sample_farthest_points(torch.from_numpy(pts_even).float()[None], torch.Tensor([N]))[0][0].numpy()
        # for debug: save some files
        # trimesh.PointCloud(pts_even, np.array([[0, 1., 0]]).repeat(len(pts_even), axis=0)).export(f'output/pclouds/eval/{name}_fps.ply')
        # pts_even = np.array(pc_even.vertices)
        # trimesh.PointCloud(pts_even, np.array([[1.0, 0., 0]]).repeat(len(pts_even), axis=0)).export(
        #     f'output/pclouds/eval/{name}_orig.ply')
        # return
        pc_count.append(len(veven))
        pc_count.append(len(vodd))
        # fscore, cd = compute_fscore(pc_odd.vertices, pc_even.vertices, thres=0.02)
        # hausd = metrics.hausdorff_distance(np.array(pc_odd.vertices), np.array(pc_even.vertices))
        fscore, cd = compute_fscore(vodd, veven, thres=0.02)
        hausd = 0.
        return {
            'Chamf': cd*self.m2cm,
            "F-score": fscore,
            'Hausdorff': hausd*self.m2cm,
        }

    # def format_output(self, args, chamfs, files_3d, fscores, hausdorffs, lpips, pc_count, pixels, psnrs, ssims):
    def format_output(self, args, errors_all, files_3d):
        test_name = args.test_name # if 'v16' not in args.identifier else 'test-v16'
        name_even = args.name_even
        name_odd = args.name_odd
        if '*' in name_even:
            name_even = sorted(glob(name_even))[0]
        if "*" in name_odd:
            name_odd = sorted(glob(name_odd))[0] # allow evaluation directly after test
        # print(f"In total {len(chamfs)} examples. even: {name_even}, odd: {name_odd}, test name {test_name}")
        # print(f"Points count: max={np.max(pc_count)}, min={np.min(pc_count)}, avg={np.mean(pc_count)}")
        keys = self.get_metric_keys()
        assert len(errors_all.keys()) == len(keys)
        es = f'In total {len(errors_all["Chamf"])} examples. even: {name_even}, odd: {name_odd}, test name {test_name}.\n'
        for k, v in errors_all.items():
            try:
                es += f'{k}: {np.mean(v):.2f}({np.std(v):.2f}) '
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
                es = f'{k}: {e}'
        print(es)
        save_dict = errors_all
        save_dict['test_name'] = test_name
        save_dict['files_3d'] = files_3d
        save_dict['name_even'] = name_even
        save_dict['name_odd'] = name_odd
        out_name = f'{osp.basename(name_even).replace("_even", "")}'
        outfile = f'results/raw/{out_name}even-odd_{args.identifier}_{str(datetime.now()).replace(":", "-").replace(" ", "-")}.pkl'
        pkl.dump(save_dict, open(outfile, 'wb'))
        # save json file as well
        dict_json = {}
        for k in save_dict.keys():
            if k in ['name_even', 'name_odd', 'files_3d', 'test_name']:
                dict_json[k] = save_dict[k]
            else:
                dict_json[k] = np.mean(save_dict[k])
        json.dump(dict_json, open(outfile.replace('.pkl', '.json').replace('/raw/', '/'), 'w'), indent=2)
        print('file saved to', outfile)

class GsplatGTEvaluator(BaseEvaluator):
    "compare against GT images"
    def evaluate(self, args):
        psnrs, ssims, lpips, pixels = [], [], [], []
        name_even = args.name_even
        scan_names = sorted(os.listdir(name_even))
        psnrs, ssims, lpips, pixels = [], [], [], []
        psnrs_train, ssims_train, lpips_train, pixels_train = [], [], [], []
        iteration = args.iteration
        for name in tqdm(scan_names):
            feven = f'{name_even}/{name}'
            files_eval = sorted(glob(osp.join(feven, f'test/ours_{iteration}/renders/*.png')))
            gt_eval = sorted(glob(osp.join(feven, f'test/ours_{iteration}/gt/*.png')))
            files_train = sorted(glob(osp.join(feven, f'train/ours_{iteration}/renders/*.png')))
            gt_train = sorted(glob(osp.join(feven, f'train/ours_{iteration}/gt/*.png')))
            reso = args.reso
            psnrs_i, ssims_i, pixels_i, lpips_i = [], [], [], []
            for odd, even in zip(files_eval, gt_eval):
                o = cv2.imread(odd)[:, :, ::-1]
                e = cv2.imread(even)[:, :, ::-1]
                pix, lpip, ssim, psnr = calc_2D_metrics(cv2.resize(o.copy(), (reso, reso)),
                                                        cv2.resize(e.copy(), (reso, reso)))
                psnrs_i.append(psnr)
                ssims_i.append(ssim)
                pixels_i.append(pix)
                lpips_i.append(lpip)
            psnrs.append(psnrs_i)
            ssims.append(ssims_i)
            pixels.append(pixels_i)
            lpips.append(lpips_i)
            psnrs_i, ssims_i, pixels_i, lpips_i = [], [], [], []
            for odd, even in zip(files_train, gt_train):
                o = cv2.imread(odd)[:, :, ::-1]
                e = cv2.imread(even)[:, :, ::-1]
                pix, lpip, ssim, psnr = calc_2D_metrics(cv2.resize(o.copy(), (reso, reso)),
                                                        cv2.resize(e.copy(), (reso, reso)))
                psnrs_i.append(psnr)
                ssims_i.append(ssim)
                pixels_i.append(pix)
                lpips_i.append(lpip)
            psnrs_train.append(psnrs_i)
            ssims_train.append(ssims_i)
            pixels_train.append(pixels_i)
            lpips_train.append(lpips_i)

        # format output
        name_even = args.name_even
        name_odd = args.name_odd
        print(f"In total {len(scan_names)} examples. even: {name_even}, odd: {name_odd}")
        # print(f"Points count: max={np.max(pc_count)}, min={np.min(pc_count)}, avg={np.mean(pc_count)}")
        print(  f"PSNR: {np.mean(psnrs):.3f}({np.std(psnrs):.2f}), "
              f"SSIM: {np.mean(ssims):.3f}({np.std(ssims):.2f}), "
              f"LPIPS: {np.mean(lpips):.3f}({np.std(lpips):.2f}), "
              f"Pixel: {np.mean(pixels):.3f}({np.std(pixels):.2f})")
        save_dict = {
            'name_even': name_even,
            "name_odd": name_odd,
            'Chamf': 0,
            "F-score": 0,
            'Hausdorff': 0,
            'PSNR': psnrs,
            'SSIM': ssims,
            'LPIPS': lpips,
            'Pixels': pixels,
            'PSNR-train': psnrs_train,
            'SSIM-train': ssims_train,
            'LPIPS-train': lpips_train,
            'Pixels-train': pixels_train,

        }
        out_name = f'{osp.basename(name_even).replace("_even", "")}'
        outfile = f'results/raw/{out_name}even-odd_{args.identifier}_{str(datetime.now()).replace(":", "-").replace(" ", "-")}.pkl'
        pkl.dump(save_dict, open(outfile, 'wb'))
        # save json file as well
        dict_json = {}
        for k in save_dict.keys():
            if k in ['name_even', 'name_odd', 'files_3d']:
                dict_json[k] = save_dict[k]
            else:
                dict_json[k] = np.mean(save_dict[k])
        json.dump(dict_json, open(outfile.replace('.pkl', '.json').replace('/raw/', '/'), 'w'), indent=2)
        print('file saved to', outfile)

if __name__ == '__main__':
    parser = BaseEvaluator.get_parser()
    args = parser.parse_args()

    evaluator = BaseEvaluator()
    evaluator.evaluate(args)
    # exit(0)

    # evaluator = GsplatGTEvaluator()
    #
    # for it in tqdm(range(1000, 30001, 1000)):
    #     args.iteration = it
    #     # args.identifier = f'error-vs-3dgs-it{it:06d}'
    #     args.identifier = f'GT-image-3dgs-vs-it{it:06d}-reso{args.reso}'
    #     evaluator.evaluate(args)










