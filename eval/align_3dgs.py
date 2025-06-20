"""
align 3dgs to one reference method using icp
"""
import json
import sys, os
sys.path.append(os.getcwd())
import trimesh
from tqdm import tqdm
from glob import glob
import numpy as np
import os.path as osp
from eval.icp_utils import reg_icp
from eval.metric_utils import normalize_percentile


class GSplatAligner:
    def __init__(self):
        pass


    def align(self, args):
        ""
        folder_ref = args.folder_tgt
        folder = args.folder_src
        if folder_ref == folder:
            print('given folders are the same, skipped.')
            return
        names = sorted(os.listdir(folder_ref))
        chamfs = []
        text = f'{osp.basename(folder).split("+")[0]}->{osp.basename(folder_ref).split("+")[0]}'
        for name in tqdm(names, desc=text):
            done = True
            for part in ['_even', '_odd']:
                outfile = osp.join(folder.replace('_even', part), name, 'transform_icp.json')
                if not osp.isfile(outfile):
                    done = False
                    break
            if done:
                continue

            files_ref = [osp.join(folder_ref.replace('_even', x), name, 'point_cloud/iteration_10000/point_cloud.ply') for x in ['_even', '_odd']]
            files = [osp.join(folder.replace('_even', x), name, 'point_cloud/iteration_10000/point_cloud.ply') for x in ['_even', '_odd']]
            full = True
            for file in files_ref + files:
                if not osp.isfile(file):
                    print(f'{file} does not exist!')
                    full = False
                    break
            if not full:
                continue
            target = np.concatenate([np.array(trimesh.load(x).vertices) for x in files_ref], 0)
            source = np.concatenate([np.array(trimesh.load(x).vertices) for x in files], 0)

            # normalize with percentile first
            mat_ref = self.get_normalize_matrix(target)
            mat = self.get_normalize_matrix(source)
            # normalize such that sv3d do not scale, but only translation
            ratio = 1. / mat_ref[0, 0]
            mat[:3] = mat[:3] * ratio
            mat_ref[:3] = mat_ref[:3] * ratio
            source = np.matmul(source, mat[:3, :3].T) + mat[:3, 3]
            target = np.matmul(target, mat_ref[:3, :3].T) + mat_ref[:3, 3]

            result_icp = reg_icp(source, target, voxel_size=0.005, scaling=True)
            transformation = np.array(result_icp.transformation)
            chamfs.append(0.)
            # save for debug
            # trimesh.PointCloud(target).export(f'output/debug/{name}_target.ply')
            # trimesh.PointCloud(source).export(f'output/debug/{name}_source.ply')

            transformation = np.matmul(transformation, mat)

            # now save this transformation
            out_dict = {
                'transformation': transformation.tolist(),
                'source': files,
                'target': files_ref,
            }
            out_ref = {
                'transformation': mat_ref.tolist(),
                'source': files_ref,
                'target': files_ref,
            }

            for part in ['_even', '_odd']:
                outfile = osp.join(folder.replace('_even', part), name, 'transform_icp.json')
                json.dump(out_dict, open(outfile, 'w'), indent=2)
                outfile = osp.join(folder_ref.replace('_even', part), name, 'transform_icp.json')
                json.dump(out_ref, open(outfile, 'w'), indent=2)
        print(f'all done')

    def get_normalize_matrix(self, target):
        vcen, size = normalize_percentile(target, 10, 0.5) # do not scale up too much
        mat = np.eye(4)
        mat[0, 0] = mat[1, 1] = mat[2, 2] = 1 / size
        mat[:3, 3] = -vcen / size
        return mat


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-ft', '--folder_tgt')
    parser.add_argument('-fs', '--folder_src')
    args = parser.parse_args()

    aligner = GSplatAligner()

    if osp.isdir(args.folder_src):
        aligner.align(args)
    else: # batch processing, faster
        folders = sorted(glob(args.folder_src))
        for folder in tqdm(folders):
            args.folder_src = folder
            aligner.align(args)