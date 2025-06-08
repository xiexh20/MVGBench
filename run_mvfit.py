"""
run 3dgs fitting from generated multi-view images
"""

import os, sys
from glob import glob
import os.path as osp

import numpy as np
from tqdm import tqdm

REDO = False

def run_combined(args):
    "all folder in one train and one render command, first 3dgs training, then render"
    if not args.white_background:
        assert '+' not in args.pat, f'please check if black background is indeed the case for {args.pat}!'
    cmd = f'python train.py --port {np.random.randint(6000, 6500)} -s "{args.pat}" -lib lgm --start {args.start}'
    if args.end is not None:
        cmd += f' --end {args.end}'
    if not args.debug:
        cmd += ' --quiet '
    if args.white_background:
        cmd += ' --white_background '
    # print(cmd)
    code = os.system(cmd)
    if code != 0:
        exit(code)

    # Now render
    cmd = f'python render.py -m "{args.pat}" --resolution 256 --elev_offset {args.elev_offset} --start {args.start}' # training gt images are resized to 256 as well
    if args.end is not None:
        cmd += f' --end {args.end}'
    if not args.debug:
        cmd += ' --quiet '
    code = os.system(cmd)
    if code != 0 :
        exit(code)

    if args.align:
        print("Aligning 3dgs")
        folder = osp.basename(osp.dirname(args.pat))
        cmd = f'python eval/align_3dgs.py -ft output/consistency/sv3dp+mvpnet50-sv3d-v21-elev030+i000_even -fs "output/consistency/{folder}even"'
        print(cmd)
        os.system(cmd)
        # re-render
        cmd = f'python render.py -m "{args.pat}" --quiet --resolution 256 --normalize_gs '
        os.system(cmd)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('pat') # eg. /home/ubuntu/efs/static/GSO30-3dgs-format/imagedream-v21-nolight_even/*
    parser.add_argument('--white_background', default=False, action='store_true') # for predictions
    parser.add_argument('-b', '--bbox_size', type=float, default=5.0)
    parser.add_argument('--align', default=False, action='store_true')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-eo', '--elev_offset', default=-10, type=float)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)

    args = parser.parse_args()

    # run_one_by_one(args)
    run_combined(args)




