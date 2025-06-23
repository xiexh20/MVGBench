"""
Evaluate the 3D consistency between two lifted 3DGS
"""
import sys, os
import cv2

sys.path.append(os.getcwd())
import os.path as osp
import numpy as np
from glob import glob
import json
from eval.eval_base import BaseEvaluator, compute_fscore
from gaussian_renderer import GaussianModel


class CovSampleEvaluator(BaseEvaluator):
    "sample points from covariance matrix and evaluate on it"
    def evaluate_3d(self, args, feven, fodd, iteration, pc_count):
        "sample points from gaussian centers"
        pcfile_even = osp.join(feven, f'point_cloud/iteration_{iteration}/point_cloud.ply')
        pcfile_odd = osp.join(fodd, f'point_cloud/iteration_{iteration}/point_cloud.ply')

        gs_odd = GaussianModel(sh_degree=3)
        gs_odd.load_ply(pcfile_odd)
        gs_even = GaussianModel(sh_degree=3)
        gs_even.load_ply(pcfile_even)

        samples_odd = self.sample_covariance(gs_odd)
        samples_even = self.sample_covariance(gs_even)

        # now downsample again
        N = 60000
        indices = np.random.choice(len(samples_odd), N, replace=len(samples_odd) < N)
        vodd = samples_odd[indices]
        indices = np.random.choice(len(samples_even), N, replace=len(samples_even) < N)
        veven = samples_even[indices]

        if args.normalize_percentile:
            nfile = f'{feven}/transform_icp.json'
            mat = np.array(json.load(open(nfile, 'r'))['transformation'])
            veven = np.matmul(veven, mat[:3, :3].T) + mat[:3, 3]
            vodd = np.matmul(vodd, mat[:3, :3].T) + mat[:3, 3]

        # compute cd and others
        pc_count.append(len(veven))
        pc_count.append(len(vodd))
        fscore, cd = compute_fscore(vodd, veven, thres=0.02)

        # now load depth and evaluate
        errors_depth = self.calc_depth_errors(feven, fodd, iteration, args.test_name)
        if errors_depth is None:
            return {} # empty dict
        return {
            'Chamf': cd * self.m2cm,
            'depth': errors_depth,
        }

    def calc_depth_errors(self, feven, fodd, iteration, test_name):
        # test_name = args.test_name
        files_odd = sorted(glob(osp.join(fodd, f'{test_name}/ours_{iteration}/depths/*.png')))
        files_even = sorted(glob(osp.join(feven, f'{test_name}/ours_{iteration}/depths/*.png')))
        if len(files_odd) != len(files_even):
            print(f'unequal number of files for {feven}({len(files_even)}) vs. {fodd}({len(files_odd)})!')
            # return 0, 1, 0.
            return None
        if len(files_odd) == 0:
            print(f'no files for {feven}({len(files_odd)})!')
            return None

        errors_depth = []
        for fe, fo in zip(files_odd, files_even):
            de, do = cv2.imread(fe, cv2.IMREAD_ANYDEPTH), cv2.imread(fo, cv2.IMREAD_ANYDEPTH)
            de, do = de * 10. / 65535, do * 10. / 65535
            mask = (do > 0) | (de > 0)
            if np.sum(mask) < 10:
                # import pdb; pdb.set_trace()
                print(f"Warning: no depth in {fo}, it is possible that 3dgs optimization failed!")
                return None

            de, do = de[mask], do[mask]
            err = np.mean(np.abs(de - do)) * self.m2cm
            errors_depth.append(err)
        return errors_depth

    def get_metrics_3d(self):
        "with depth"
        return ["Chamf", 'depth']

    def sample_covariance(self, gs_odd: GaussianModel, rounds=3):
        cov = gs_odd.get_covariance(strip=False).detach().cpu().numpy()
        try:
            chol_decompositions = np.linalg.cholesky(cov)
        except:
            epsilon = 1e-6  # Small positive value to have positive definite matrix
            cov = cov + np.eye(3)[None] * epsilon
            chol_decompositions = np.linalg.cholesky(cov)
        centers = gs_odd._xyz.detach().cpu().numpy()
        samples_odd = [centers]
        for i in range(rounds):
            samples = np.random.randn(len(centers), 3)
            samples = centers + np.einsum('ijk,ik->ij', chol_decompositions, samples)
            samples_odd.append(samples)
        samples_odd = np.concatenate(samples_odd, axis=0)
        return samples_odd






if __name__ == '__main__':
    parser = BaseEvaluator.get_parser()
    args = parser.parse_args()

    evaluator = CovSampleEvaluator()
    evaluator.evaluate(args)