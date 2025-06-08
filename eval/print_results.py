"""
format results into a table, avoid tedious formating
"""

import sys, os, json
import pickle as pkl
from glob import glob
import os.path as osp

import numpy as np


def print_results(args):

    plus_minus = '\u00B1'
    up_arrow = "\u2191"
    down_arrow = "\u2193"

    pkl_files = [
        'results/sv3dp+sv3d-v21-nolighteven-odd_val-random-v16-nolight_2024-11-05-00-02-49.320223.json',
        'results/cape-lvis-cosine-lr+sv3d-v21-nolight+step-009900even-odd_None_2024-11-01-15-31-39.734025.json',
        'results/cape-lvis-bs64+sv3d-v21-nolight+step-010000-mergedeven-odd_None_2024-11-03-19-46-47.964783.json',
        'results/cape-lvis-dyna+sv3d-v21-nolight+step-010000-mergedeven-odd_None_2024-11-04-15-38-35.453943.json',
        'results/cape-conv-static-80k+sv3d-v21-nolight+step-005000-mergedeven-odd_None_2024-11-04-22-28-07.122615.json'
    ]

    pkl_files = sorted(glob('results/raw/syncdreamer+mvdfusion-v16-frame00001*elevIn*even*.pkl'))
    legends = [f'elev {x}' for x in [-10, 0, 15, 30, 40]]

    pkl_files = [
        'results/raw/syncdreamer+mvdfusion-v16-frame00001even-odd_None_2024-10-29-09-45-57.385787.pkl',
        'results/raw/eschernet+zero123-v21-frame00001+i000even-odd_None_2024-10-29-09-37-16.228261.pkl',
        'results/raw/sv3dp+sv3d-v21-frame00001+i000even-odd_None_2024-10-29-12-14-12.133497.pkl',
        'results/raw/cape-lvis-bs64+sv3d-v21-frame00001+step-010000-mergedeven-odd_None_2024-11-03-22-38-14.145518.pkl'
    ]

    pkl_files = [
        'results/raw/sv3dp+sv3d-v21-nolighteven-odd_val-random-v16-nolight_2024-11-05-00-02-49.320223.pkl',
        'results/cape-lvis-bs64+sv3d-v21-nolight+step-010000-mergedeven-odd_None_2024-11-03-19-46-47.964783.json',
        'results/cape-lvis-dyna+sv3d-v21-nolight+step-010000-mergedeven-odd_None_2024-11-04-15-38-35.453943.json',
        'results/raw/cape-conv-static-80k+sv3d-v21-nolight+step-010000-mergedeven-odd_None_2024-11-05-16-40-23.393715.pkl',
        'results/raw/cape-conv-test+sv3d-v21-nolight+step000000even-odd_None_2024-11-05-21-51-50.995061.pkl'
        
        # 'results/raw/sv3dp+sv3d-v21-frame00001+i000even-odd_None_2024-10-29-12-14-12.133497.pkl',
        # 'results/raw/cape-lvis-bs64+sv3d-v21-frame00001+step-010000-mergedeven-odd_None_2024-11-03-22-38-14.145518.pkl',
        # 'results/raw/cape-lvis-dyna+sv3d-v21-frame00001+step-010000-mergedeven-odd_None_2024-11-04-18-30-42.106637.pkl',
        # 'results/raw/cape-conv-static-80k+sv3d-v21-frame00001+step-010000-mergedeven-odd_None_2024-11-05-16-48-03.339638.pkl'
    ]
    legends = ['+GSO'] * 4 + ['+real']*4
    # legends = [''] * len(pkl_files)

    pkl_files = sorted(glob('results/*error-vs-Inelev-reso256*'))
    legends = [f'+elev{x}' for x in [0, 15, 30, 45]*2]

    # with random focal length
    # pkl_files = [
    #     # 'results/raw/sv3d-v21-nolight+random-focaleven-odd_None_2024-11-12-00-49-28.894301.pkl',
    #     # 'results/raw/zero123-v21-nolight+random-focaleven-odd_None_2024-11-12-00-49-50.810084.pkl',
    #     'results/raw/sv3dp+sv3d-v21-nolighteven-odd_error-vs-3dgs-it010000_2024-10-23-08-57-58.110825.pkl',
    #     'results/raw/eschernet+zero123-v21-nolighteven-odd_error-vs-3dgs-it010000_2024-10-23-19-08-16.410737.pkl',
    #     'results/raw/cape-conv-static-80k+sv3d-v21-nolight+random-focal+step-021000-mergedeven-odd_None_2024-11-12-00-38-42.272597.pkl',
    #     'results/raw/eschernet+zero123-v21-nolight+random-focal+i000even-odd_None_2024-11-11-23-56-58.196304.pkl'
    # ]
    # legends = [' original', ' original', ' cropped', ' cropped']

    # new filtered results with 1.0 3dgs bbox
    # pkl_files =[
    #     'results/raw/sv3d-v21-nolight-bbox2.6even-odd_3dgs-bbox2.6-no-bf_2024-11-12-11-31-11.180919.pkl',
    #     'results/raw/imagedream-v21-nolight-bbox2.6even-odd_3dgs-bbox2.6-no-bf_2024-11-12-11-42-40.556726.pkl',
    #     'results/raw/zero123-v21-nolight-bbox2.6even-odd_3dgs-bbox2.6-no-bf_2024-11-12-11-36-59.504746.pkl',
    #     'results/old/mvdfusion-v16-elev030-amb1.0even-odd_gt-nolight_2024-10-22-08-29-09.494200.json',
    #     'results/raw/sv3d-v21-nolighteven-odd_None_2024-11-15-11-20-42.041867.pkl',
    #     'results/raw/imagedream-v21-nolighteven-odd_None_2024-11-15-11-21-36.169732.pkl',
    #     'results/raw/zero123-v21-nolighteven-odd_3dgs-bbox2.0-bfilter1.0_2024-11-12-11-16-51.367747.pkl', # TODO： is this really that bad？
    #     'results/raw/mvdfusion-v21-nolighteven-odd_None_2024-11-15-11-14-58.234034.pkl'
    # ]
    # legends = [' bound 2.6'] * 4 + [' bound 2.0'] * 4
    #
    # # filtered results with prediction
    # pkl_files = [
    #     'results/old/raw/eschernet+zero123-v21-nolighteven-odd_mvd-nolight-reso256_2024-10-08-06-45-21.318971.pkl',
    #     'results/old/raw/mvdfusion+mvdfusion-v16-elev030-amb1.0even-odd_mvd-nolight-reso256_2024-10-22-08-27-39.315815.pkl',
    #     'results/old/raw/sv3dp+sv3d-v21-nolighteven-odd_mvd-nolight-reso256_2024-10-08-06-58-16.850470.pkl',
    #     'results/old/raw/syncdreamer+mvdfusion-v16-nolighteven-odd_mvd-nolight-reso256_2024-10-22-08-55-57.039935.pkl'
    #     # 'results/raw/sv3dp+sv3d-v21-nolighteven-odd_3dgs-bbox2.0-bfilter1.0_2024-11-12-12-50-47.783907.pkl',
    #     # 'results/raw/eschernet+zero123-v21-nolighteven-odd_3dgs-bbox2.0-bfilter1.0_2024-11-12-12-51-05.238371.pkl',
    # ] + sorted(glob('results/*mvd-optimal*'))
    # legends = [' bbox 2.6'] * 4 + [' bbox 2.0'] * 4
    #
    # # filter + re-render
    # pkl_files = [
    #     'results/raw/sv3dp+sv3d-v21-elev030even-odd_3dgs-bbox2.0-bfilter1.0_2024-11-12-09-54-25.198692.pkl',
    #     'results/raw/sv3dp+sv3d-v21-elev030-bbox2.6even-odd_3dgs-bbox2.6-bfilter1.0-rerender_2024-11-12-13-01-55.991352.pkl'
    # ]
    # legends = [' bound 2.0 fit', ' b2.6 + filter']

    # fine tune on the week of Nov4-11
    # pkl_files = [
    #     'results/sv3dp+sv3d-v21-nolighteven-odd_val-random-v16-nolight_2024-11-05-00-02-49.320223.json',
    #     'results/cape-lvis-bs64+sv3d-v21-nolight+step-010000-mergedeven-odd_None_2024-11-03-19-46-47.964783.json',
    #     'results/cape-prog-lgm+sv3d-v21-nolight+step-006000-mergedeven-odd_None_2024-11-07-18-40-13.632504.json',
    #     'results/cape-conv-static-80k+sv3d-v21-nolight+step-021000-mergedeven-odd_None_2024-11-11-19-18-01.412774.json',
    #     'results/cape-conv-dyna-80k+sv3d-v21-nolight+step-012000-mergedeven-odd_None_2024-11-10-16-45-28.409176.json'
    # ]
    # legends = [''] * len(pkl_files)

    pkl_files = [
        'results/sv3dp+sv3d-v21-nolight-debugeven-odd_3dgs-bbox2.0-no-bf_2024-11-13-10-29-58.611774.json',
        'results/eschernet+zero123-v21-nolighteven-odd_3dgs-bbox2.0-bfilter1.0_2024-11-12-11-11-55.111876.json',
        'results/sv3dp+sv3d-v21-nolight+random-focal+i000even-odd_None_2024-11-13-18-09-07.941908.json',
        'results/eschernet+zero123-v21-nolight+random-focal+i000even-odd_None_2024-11-13-16-37-46.595828.json',
        'results/eschernet+random-focal-v21+i000even-odd_None_2024-11-13-01-12-28.510363.json',
        'results/sv3dp+random-focal-v21+i000even-odd_None_2024-11-13-21-42-29.440758.json'
    ]
    legends = [' original', ' original', ' re-cropped', ' recropped', ' rfocal', ' rfocal']

    # mixed evaluation: each method with different rendering setup
    pkl_files = sorted(glob('results/raw/*frame*elev030*mixed-eval*')) + [
        'results/sv3dp+random-focal-v21+i000even-odd_None_2024-11-13-21-42-29.440758.json',
        'results/eschernet+random-focal-v21+i000even-odd_None_2024-11-13-01-12-28.510363.json'
    ]
    legends = ['+' + osp.basename(x).split('+')[1][:7] for x in pkl_files]

    # pkl_files = sorted(glob('results/*norm-qp10-new-render-scale0.5*.json'))
    # # pkl_files = sorted(glob('results/*frame*elev*norm-qp10-new-render-scale0.5*.json'))
    # # legends = ['+' + osp.basename(x).split('+')[1][9:17] for x in pkl_files]
    # legends = ['+' + osp.basename(x).split('+')[1][:3] for x in pkl_files]
    # legends = [leg + '' if 'elev030' not in x else leg+'+e30' for leg, x in zip(legends, pkl_files)]
    #
    # pkl_files = sorted(glob('results/*frame*elev*_no-norm_*.json')) + ['results/sv3dp+sv3d-v21-frame00001-elev030+i000even-odd_mixed-eval-elev030_2024-11-15-19-54-05.002655.json']
    # pkl_files = sorted(glob('results/*mvpnet*elev*_norm-icp-rerender_*.json')) + sorted(glob('results/*mvpnet*manual*_norm-icp-rerender-rot_*.json'))
    # pkl_files = [x for x in pkl_files if 'step' not in x]
    # # legends = ['+' + osp.basename(x).split('+')[1][:6] for x in pkl_files]
    # legends = ['+elev30' if 'manual' not in x else '+manual' for x in pkl_files]
    # legends = ['']*len(pkl_files)
    # legends = [leg + '' if 'recrop' not in x else leg + '+recrop' for leg, x in zip(legends, pkl_files)]

    # pkl_files = sorted(glob('results/cape*_norm-p10_*.json')) #+ sorted(glob('results/sv3dp-lvis*_no-norm_*.json'))
    # # pkl_files = sorted(glob('results/*+sv3d*_norm-p10_*.json')) + sorted(glob('results/*+mvpnet*_norm-p10_*.json'))
    # legends = ['+' + osp.basename(x).split('+')[1][:7] for x in pkl_files]

    # summary of fine-tuned models
    # pkl_files = [
    #     'results/sv3dp+sv3d-v21-nolighteven-odd_mvd-optimal_2024-11-15-23-56-53.687090.json',
    #     'results/cape-lvis-bs64+sv3d-v21-nolight+step-010000-mergedeven-odd_3dgsv2_2024-11-19-20-20-05.068254.json',
    #     'results/cape-prog-lgm+sv3d-v21-nolight+step-006000-mergedeven-odd_3dgsv2_2024-11-19-20-31-23.461270.json',
    #     'results/cape-conv-static-80k+sv3d-v21-nolight+step-078000-mergedeven-odd_3dgsv2_2024-11-19-20-10-19.185740.json',
    #     'results/sv3dp-lvis-static+sv3d-v21-nolight+step-010500-mergedeven-odd_None_2024-11-13-12-36-42.351772.json'
    # ]
    # legends = ['']* len(pkl_files)
    pkl_files = [
        'results/sv3dp+sv3d-v21-nolighteven-odd_mvd-optimal_2024-11-15-23-56-53.687090.json',
        'results/cape-conv-static-80k+sv3d-v21-nolight+step-078000-mergedeven-odd_3dgsv2_2024-11-19-20-10-19.185740.json',
        'results/sv3dp-lvis-static+sv3d-v21-nolight+step-010500-mergedeven-odd_None_2024-11-13-12-36-42.351772.json',
        'results/sv3dp+gso100-sv3d-v21-elev012-amb1.0+i000even-odd_None_2024-11-19-08-47-33.932701.json',
        'results/cape-conv-static-80k+gso100-sv3d-v21-elev012-amb1.0+step-078000-mergedeven-odd_None_2024-11-19-08-43-47.616415.json',
        'results/sv3dp-lvis-static+gso100-sv3d-v21-elev012-amb1.0+step-010500-mergedeven-odd_None_2024-11-19-18-17-00.194460.json',
        'results/sv3dp+omni202-sv3d-v21-elev012-amb1.0+i000even-odd_None_2024-11-19-08-47-41.223850.json',
        'results/cape-conv-static-80k+omni202-sv3d-v21-elev012-amb1.0+step-078000-mergedeven-odd_None_2024-11-19-08-46-59.609110.json',
        'results/sv3dp-lvis-static+omni202-sv3d-v21-elev012-amb1.0+step-010500-mergedeven-odd_None_2024-11-19-19-52-27.152159.json',
    ]
    pkl_files = [
        'results/sv3dp+sv3d-v21-nolighteven-odd_cov-sample-60k_2024-11-27-02-07-51.074879.json',
        'results/raw/cape-conv-static-80k+sv3d-v21-nolight+step-078000-mergedeven-odd_cov-sample-60k_2024-11-27-02-30-26.372376.pkl',
        'results/sv3dp-lvis-static+sv3d-v21-nolight+step-010500-mergedeven-odd_cov-sample-60k_2024-11-27-02-08-16.250727.json',
        'results/sv3dp+gso100-sv3d-v21-elev012-amb1.0+i000even-odd_cov-sample-60k_2024-11-27-01-51-09.355052.json',
        'results/cape-conv-static-80k+gso100-sv3d-v21-elev012-amb1.0+step-078000-mergedeven-odd_cov-sample-60k_2024-11-27-00-16-55.011759.json',
        'results/sv3dp-lvis-static+gso100-sv3d-v21-elev012-amb1.0+step-010500-mergedeven-odd_cov-sample-60k_2024-11-27-00-16-56.028738.json',
        'results/sv3dp+omni202-sv3d-v21-elev012-amb1.0+i000even-odd_cov-sample-60k_2024-11-27-01-52-06.181284.json',
        'results/raw/cape-conv-static-80k+omni202-sv3d-v21-elev012-amb1.0+step-078000-mergedeven-odd_cov-sample-60k_2024-11-27-01-57-54.979794.pkl',
        'results/sv3dp-lvis-static+omni202-sv3d-v21-elev012-amb1.0+step-010500-mergedeven-odd_cov-sample-60k_2024-11-27-00-22-36.888359.json'

    ]
    # pkl_files = [
    #     'results/sv3dp+sv3d-v21-nolighteven-odd_dmap-skip2_2024-11-27-02-24-08.775245.json',
    #     'results/cape-conv-static-80k+sv3d-v21-nolight+step-078000-mergedeven-odd_dmap-skip2_2024-11-27-02-26-05.823846.json',
    #     'results/sv3dp-lvis-static+sv3d-v21-nolight+step-010500-mergedeven-odd_dmap-skip2_2024-11-27-02-24-03.545433.json',
    #     'results/sv3dp+gso100-sv3d-v21-elev012-amb1.0+i000even-odd_dmap-skip2_2024-11-27-01-47-02.760084.json',
    #     'results/cape-conv-static-80k+gso100-sv3d-v21-elev012-amb1.0+step-078000-mergedeven-odd_dmap-skip2_2024-11-27-00-37-59.350465.json',
    #     'results/sv3dp-lvis-static+gso100-sv3d-v21-elev012-amb1.0+step-010500-mergedeven-odd_dmap-skip2_2024-11-27-00-41-31.035548.json',
    #     'results/sv3dp+omni202-sv3d-v21-elev012-amb1.0+i000even-odd_dmap-skip2_2024-11-27-01-46-37.197064.json',
    #     'results/cape-conv-static-80k+omni202-sv3d-v21-elev012-amb1.0+step-078000-mergedeven-odd_dmap-skip2_2024-11-27-01-44-32.060632.json',
    #     'results/sv3dp-lvis-static+omni202-sv3d-v21-elev012-amb1.0+step-010500-mergedeven-odd_dmap-skip2_2024-11-27-01-43-28.410355.json',
    #
    # ]
    # legends = ['+' + osp.basename(x).split('+')[1][:7] for x in pkl_files]

    # pkl_files = sorted(glob('results/raw/cape-conv-kiui+sv3d-v21-nolight+step*re-render*.pkl'))
    # pkl_files = sorted(glob('results/raw/*_re-render256-resample_*.pkl'))
    # pkl_files = [
    #     'results/cape-conv-static-80k+gso100-sv3d-v21-elev012-amb1.0+step-078000-mergedeven-odd_re-render256-resample_2024-12-03-20-00-12.001768.json',
    #     'results/cape-conv-kiui+gso100-sv3d-v21-elev012-amb1.0+step-019500-mergedeven-odd_re-render256-resample_2024-12-03-19-56-33.167097.json',
    #     'results/cape-conv-static-80k+omni202-sv3d-v21-elev012-amb1.0+step-078000-mergedeven-odd_re-render256-resample_2024-12-03-19-31-51.931627.json',
    #     'results/cape-conv-kiui+omni202-sv3d-v21-elev012-amb1.0+step-019500-mergedeven-odd_re-render256-resample_2024-12-03-19-17-22.094671.json'
    # ]
    # legends = [' 80k obj + GSO100', ' 150k obj + GSO100', ' 80k obj + Omni202', ' 150k obj + Omni202']
    # legends = ['+' + osp.basename(x).split('+')[2][:8] for x in pkl_files]
    # pkl_files = [
    #     'results/raw/cape-conv-static-80k+mvpnet50-sv3d-v21-elev030+step-078000-mergedeven-odd_re-render256_2024-12-03-22-41-23.673194.pkl',
    #     'results/raw/cape-conv-kiui+mvpnet50-sv3d-v21-elev030+step-050000-mergedeven-odd_re-render256_2024-12-03-22-39-56.797560.pkl'
    # ]
    # legends = ['80k obj + MVPNet', '150k obj + MVPNet']

    pkl_files = [
        'results/raw/sv3dp+mvpnet50-sv3d-v21-elev030+i000even-odd_ re-render256-resample_2024-12-03-23-14-46.280391.pkl',
        'results/raw/sv3dp+mvpnet50-sv3d-v21-manual+i000even-odd_re-render256-resample_2024-12-03-23-08-27.797139.pkl'

    ]
    legends = ['all elev 30', 'manual annotation']

    pkl_files = [
        # 'results/raw/sv3dp+mvpnet50-sv3d-v21-elev030+i000even-odd_ re-render256-resample_2024-12-03-23-14-46.280391.pkl',
        'results/raw/sv3dp+mvpnet50-sv3d-v21-manual-elev030+i000even-odd_re-render256-resample_2024-12-06-19-34-50.468667.pkl',
        'results/raw/sv3dp+mvpnet50-sv3d-v21-manual+i000even-odd_re-render256-resample_2024-12-03-23-08-27.797139.pkl',
        # 'results/raw/syncdreamer+mvpnet50-mvdfusion-v16-elev030even-odd_re-render256-resample-align_2024-12-05-17-33-42.198562.pkl',
        'results/raw/syncdreamer+mvpnet50-mvdfusion-v16-manual-elev030even-odd_render256-align_2024-12-09-11-08-19.768502.pkl',
        'results/raw/syncdreamer+mvpnet50-mvdfusion-v16-manualeven-odd_re-render256-resample-align_2024-12-05-17-04-11.610665.pkl',
        # 'results/raw/eschernet+mvpnet50-zero123-v21-elev030+i000even-odd_re-render256-resample-align_2024-12-05-17-35-16.583079.pkl',
        'results/raw/eschernet+mvpnet50-zero123-v21-manual-elev030+i000even-odd_render256-align_2024-12-09-11-08-47.005955.pkl',
        'results/raw/eschernet+mvpnet50-zero123-v21-manual+i000even-odd_re-render256-resample-align_2024-12-05-17-09-27.229626.pkl',
        'results/raw/free3d+mvpnet50-zero123-v21-manual-elev030+i000even-odd_render256-align_2024-12-09-14-53-16.638286.pkl',
        'results/raw/free3d+mvpnet50-zero123-v21-manual+i000even-odd_render256-align_2024-12-09-14-52-33.674271.pkl',
        'results/raw/epidiff+mvpnet50-mvdfusion-v16-manual-elev030+i000even-odd_render256-align_2024-12-09-14-56-36.037646.pkl',
        'results/raw/epidiff+mvpnet50-mvdfusion-v16-manual+i000even-odd_render256-align_2024-12-09-14-56-56.209760.pkl'
    ] # TODO: check qualitative examples. or print statistics
    legends = ['SV3D elev30 same', 'SV3D manual',
               'syncdr elev30 same','syncdreamer manual',
               'escher elev30 same', 'eschernet manual',
               'free3d elev30 same', 'free3d manual',
               'epidiff elev30 same', 'epidiff manual',
               ]

    # pkl_files = sorted(glob('results/*test-fid-all-images*'))
    # legends = ['+' + osp.basename(x).split('+')[1][:8] for x in pkl_files]]
    pkl_files = sorted(glob('results/*mvpnet50-manualv2-full-cov*.json') )
    pkl_files = [x for x in pkl_files if 'step-0195' not in x and 'step-0445' not in x and 'step-0105' not in x and 'step-078' not in x]
    legends = [''] * len(pkl_files)
    #
    # # ICP alignment, use rotation or not? not much difference! for simplicity, scale only
    # pkl_files = sorted(glob('results/*mixed-eval-align-rot*.json'))
    # legends = ['+' + osp.basename(x).split('+')[1][:8] for x in pkl_files]
    #
    # # 3dgs resample, use 1std or full covariance?
    # pat='resample-full-cov'
    # pkl_files = sorted(glob(f'results/*nolight*_{pat}_*.json')) + sorted(glob(f'results/*gso100*_{pat}_*.json')) + sorted(glob(f'results/*omni202*_{pat}_*.json'))
    # pat = 'optimal-new-fid'
    # pkl_files = sorted(glob(f'results/*gso100*_{pat}_*.json'))
    # pkl_files = sorted(glob(f'results/*sv3dp+*error-vs-elev+-15_*.json'))

    # different architecture
    pkl_files = [
        'results/sv3dp+gso100-sv3d-v21-elev012-amb1.0+i000even-odd_random-v4_2025-01-05-19-44-13.986829.json',
        # 'results/sv3dp-lvis-static+gso100-sv3d-v21-elev012-amb1.0+i000+step-017000-mergedeven-odd_random-v4_2025-01-05-20-58-59.361576.json',
        'results/sv3dp-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+step-028000-mergedeven-odd_random-v4_2025-01-31-17-30-48.553118.json',
        'results/rcn-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+step-026000-mergedeven-odd_random-v4_2025-01-19-16-04-36.344389.json',
        'results/raw/cape-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+step-026500-mergedeven-odd_random-v4_2025-01-07-22-09-16.643172.pkl',
        'results/rcn-cape+gso100-sv3d-v21-elev012-amb1.0+i000+step-021500-mergedeven-odd_random-v4_2025-01-20-21-19-59.768069.json',
        'results/cape-dino-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+step-050000-mergedeven-odd_random-v4_2025-01-18-17-27-17.627459.json',
        'results/cape-conv-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+step-050000-mergedeven-odd_random-v4_2025-01-05-19-46-20.270859.json',
        'results/rcn-conv-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+step-050000-mergedeven-odd_random-v4_2025-02-07-23-38-45.425416.json',
        'results/cape-conv-syncv2blend+gso100-sv3d-v21-elev012-amb1.0+i000+step-006300-mergedeven-odd_random-v4_2025-01-06-16-01-11.876669.json'
    ]
    # pkl_files = [
    #     'results/sv3dp+omni202-sv3d-v21-elev012-amb1.0+i000even-odd_random-v4-omni202_2025-01-06-12-33-43.273869.json',
    #     'results/sv3dp-lvis-static+omni202-sv3d-v21-elev012-amb1.0+step-017000-mergedeven-odd_random-v4-omni202_2025-01-06-12-47-11.309316.json',
    #     'results/cape-conv-kiui+omni202-sv3d-v21-elev012-amb1.0+step-050000-mergedeven-odd_random-v4-omni202_2025-01-06-11-32-05.315260.json',
    #     'results/cape-conv-syncv2blend+omni202-sv3d-v21-elev012-amb1.0+i000+step-006300-mergedeven-odd_random-v4-omni202_2025-01-07-17-02-45.624021.json'
    # ]
    # pkl_files = [
    #     'results/sv3dp+co3d2seq-sv3d-v21-manual+i000even-odd_align-elev+-15_2025-01-07-01-49-17.193645.json',
    #     'results/sv3dp-lvis-static+co3d2seq-sv3d-v21-manual+i000+step-017000-mergedeven-odd_align-elev+-15_2025-01-07-02-03-50.785505.json',
    #     'results/cape-conv-kiui+co3d2seq-sv3d-v21-manual+i000+step-050000-mergedeven-odd_align-elev+-15_2025-01-07-17-49-21.523905.json',
    #     'results/cape-conv-syncv2blend+co3d2seq-sv3d-v21-manual+i000+step-006300-mergedeven-odd_align-elev+-15_2025-01-07-11-25-11.604854.json'
    # ]
    # legends = ['+' + osp.basename(x).split('+')[1][:1] for x in pkl_files]
    legends = ['sv3dp', 'sv3dp-tune', 'sv3d+RCN', 'sv3d+CaPE', 'sv3d+RCN+CaPE', 'sv3d+CaPE+DINO', 'sv3d+CaPE+Conv', 'sv3d+RCN+Conv', 'sv3d+CaPE+Conv+Sync']
    # legends = ['sv3dp', 'sv3dp-tune', 'sv3d+CaPE+Conv', 'sv3d+CaPE+Conv+Sync']

    # train as test
    # pkl_files = [
    #     'results/white-gt+train-as-test-sv3deven-odd_random-v4_2025-02-17-13-22-38.063925.json',
    #     'results/rcn-conv-kiui+train-as-test-sv3d+i000+step-050000-mergedeven-odd_random-v4_2025-02-17-11-19-56.900854.json',
    #     'results/white-gt+sv3d-reso576even-odd_random-v4_2025-02-18-19-27-52.955801.json',
    #     'results/rcn-conv-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+step-050000-mergedeven-odd_random-v4_2025-02-07-23-38-45.425416.json',
    #     'results/rcn-conv-kiui+co3d2seq-sv3d-v21-manual+i000+step-050000-mergedeven-odd_align-elev+-15_2025-02-07-19-24-11.272623.json',
    #     'results/rcn-conv-kiui+mvimgnet230-sv3d-v21-manual+i000+step-050000-mergedeven-odd_align-elev+-15_2025-02-08-03-51-33.400295.json'
    #
    # ]
    # legends = ['objav-GT', 'objav-pred', 'GSO100-GT', 'GSO100-pred', 'CO3D', 'MVImgnet']

    # pkl_files = [
    #     'results/eschernet+gso100-zero123-v21-elev000-amb1.0+i000even-odd_random-v4_2025-01-05-19-49-10.738483.json',
    #     'results/sv3dp+gso100-sv3d-v21-elev012-amb1.0+i000even-odd_random-v4_2025-01-05-19-44-13.986829.json',
    #     'results/cape-conv-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+step-050000-mergedeven-odd_random-v4_2025-01-05-19-46-20.270859.json',
    #     'results/eschernet+gso100-zero123-v21-elev000-amb1.0+i000-rtrajeven-odd_random-v4_2025-01-28-20-16-03.518753.json',
    #     'results/raw/sv3dp+gso100-sv3d-v21-elev012-amb1.0+i000+rtrajeven-odd_random-v4_2025-01-28-20-14-49.952095.pkl',
    #     'results/raw/cape-conv-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+rtraj+step-050000-mergedeven-odd_random-v4_2025-01-28-21-35-16.529452.pkl'
    # ]
    # legends = ['EscherNet', 'SV3D', 'Ours', 'EscherNet new traj', 'SV3D new traj', 'Ours new traj']

    # pkl_files = [
    #     'results/sv3dp+gso100-sv3d-v21-elev012-amb1.0+i000even-odd_random-v4_2025-01-05-19-44-13.986829.json',
    #     'results/sv3dp-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+step-018750-mergedeven-odd_random-v4_2025-01-27-13-35-17.405553.json',
    #     'results/rcn-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+step-026000-mergedeven-odd_random-v4_2025-01-19-16-04-36.344389.json',
    #     'results/cape-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+step-026500-mergedeven-odd_random-v4_2025-01-07-22-09-16.643172.json',
    #     'results/rcn-cape+gso100-sv3d-v21-elev012-amb1.0+i000+step-021500-mergedeven-odd_random-v4_2025-01-20-21-19-59.768069.json'
    # ]
    # legends = ['sv3d', 'sv3d-tune', 'sv3d+RCN', 'sv3d+CaPE', 'sv3d+CaPE+RCN']

    # pkl_files = [
    #     'results/raw/sv3dp+sv3d-v21-nolighteven-odd_random-v4-no-depth-norm_2025-01-08-00-47-13.630875.pkl',
    #     'results/cape-conv-kiui+sv3d-v21-nolight+step-050000-mergedeven-odd_random-v4_2024-12-30-13-59-54.157802.json',
    #     'results/cape-conv-syncv2blend+sv3d-v21-nolight+i000+step-006300-mergedeven-odd_random-v4_2025-01-05-22-19-42.298942.json',
    #     'results/sv3d-v21-nolighteven-odd_random-v4-no-dnorm_2024-12-27-14-20-44.248265.json',
    #
    # ]
    # legends = ['sv3dp', 'sv3d+CaPE+Conv','sv3d+CaPE+Conv+Sync', 'GT images', ]
    # pkl_files = [
    #     'results/cape-kiui+sv3d-v21-nolight+i000+step-026500-mergedeven-odd_random-v4_2025-01-03-09-55-48.769890.json',
    #     'results/cape-conv-kiui+sv3d-v21-nolight+step-050000-mergedeven-odd_random-v4_2024-12-30-13-59-54.157802.json',
    #     'results/cape-dino-kiui+sv3d-v21-nolight+i000+step-021500-mergedeven-odd_random-v4_2025-01-14-11-17-43.211459.json'
    # ]
    # legends = ['SV3D+CaPE+CLIP', 'SV3D+CaPE+Conv', 'SV3D+CaPE+DINO']
    # # pkl_files = sorted(glob('results/cape-dino-kiui+sv3d*.json'))
    # pkl_files = sorted(glob('results/cape-conv-syncv2blend-slow+sv3d-v21-nolight+i000+step-0*'))
    # pkl_files = ['results/cape-conv-kiui+sv3d-v21-nolight+step-050000-mergedeven-odd_random-v4_2024-12-30-13-59-54.157802.json'] + pkl_files
    # legends = ['cape-conv-step50k'] + [osp.basename(x).split('+')[0][:13] + '+' + osp.basename(x).split('+')[3][6:12] for x in pkl_files[1:]]

    pkl_files = [
        # 'results/sv3dp+co3d2seq-sv3d-v21-manual+i000+neweleveven-odd_align-elev+-15_2025-02-25-23-42-55.723514.json',
        # 'results/sv3dp+co3d2seq-sv3d-v21-manual+i000even-odd_align-elev+-15_2025-01-07-01-49-17.193645.json',
        # 'results/rcn-conv-kiui+co3d2seq-sv3d-v21-manual+i000+step-050000-merged+neweleveven-odd_align-elev+-15_2025-02-24-00-31-52.317456.json',
        # 'results/rcn-conv-kiui+co3d2seq-sv3d-v21-manual+i000+step-050000-mergedeven-odd_align-elev+-15_2025-02-07-19-24-11.272623.json',
        'results/sv3dp+mvimgnet230-sv3d-v21-manual+i000+neweleveven-odd_align-elev+-15_2025-02-23-12-29-32.788261.json',
        'results/sv3dp+mvimgnet230-sv3d-v21-manual+i000even-odd_align-elev+-15_2025-01-13-13-55-30.049673.json',
        'results/rcn-conv-kiui+mvimgnet230-sv3d-v21-manual+i000+step-050000-merged+neweleveven-odd_align-elev+-15_2025-02-23-13-09-10.365122.json',
        'results/rcn-conv-kiui+mvimgnet230-sv3d-v21-manual+i000+step-050000-mergedeven-odd_align-elev+-15_2025-02-08-03-51-33.400295.json',
    ]
    legends = ['12345 elev+sv3d', 'our elev+sv3d', '12345 elev+our model', 'our elev+our model']

    # downsample-sv3d results
    pkl_files = [
        'results/syncdreamer+mvdfusion-v16-elev030-amb1.0+i000even-odd_random-v4-light-round1_2025-01-06-14-34-49.388848.json',
        'results/sv3dp+sv3d-v21-nolighteven-odd_random-v4-no-depth-norm_2025-01-08-00-47-13.630875.json',
        'results/sv3dp+sv3d-v21-nolight+v16even-odd_random-v4-down-v16_2025-02-27-12-51-03.839242.json',
        'results/rcn-conv-kiui+sv3d-v21-nolight+i000+step-050000-mergedeven-odd_random-v4_2025-02-07-00-34-38.355294.json',
        'results/rcn-conv-kiui+sv3d-v21-nolight+i000+step-050000-merged+v16even-odd_random-v4-down-v16_2025-02-27-12-47-50.871584.json'
    ]
    legends = ['syncdreamer 16views', 'sv3d 20 views', 'sv3d 16 views', 'ours 20 views', 'ours 16 views']

    # same elevation, but different camera focal and distance
    pkl_files = [
        'results/mvdfusion-v21-nolighteven-odd_random-v4-real_2025-03-01-12-13-27.585240.json',
        'results/raw/sv3d-v21-elev030even-odd_random-v4-v16_2025-03-01-14-58-36.804149.pkl',
        'results/raw/zero123-v21-elev030even-odd_random-v4-v16_2025-03-01-14-59-49.190668.pkl',
        'results/mvdfusion-v21-nolighteven-odd_random-v16-fov45-d2.8-3.5_2025-03-01-16-40-57.912994.json',
        'results/sv3d-v21-elev030even-odd_random-v16-fov45-d2.8-3.5_2025-03-01-16-43-06.370539.json',
        'results/zero123-v21-elev030even-odd_random-v16-fov45-d2.8-3.5_2025-03-01-16-47-31.832738.json'
        # 'results/white-gt+mvdfusion-v21-nolighteven-odd_random-v4-white-gt_2025-03-01-13-20-27.054042.json',
        # 'results/raw/white-gt+sv3d-v21-elev030even-odd_random-v4-white-gt_2025-03-01-15-39-46.024497.pkl',
        # 'results/raw/white-gt+zero123-v21-elev030even-odd_random-v4-white-gt_2025-03-01-15-34-13.772690.pkl'
    ]
    legends = [osp.basename(x)[:20] for x in pkl_files]

    # overlap views split
    pkl_files = [
        'results/mvdfusion-v16-elev030-amb1.0even-odd_random-v4-black-gt_2025-02-27-11-23-49.270418.json',
        'results/v3d-v18-elev000-amb1.0even-odd_random-v4-black-gt_2025-02-27-11-35-56.447172.json',
        'results/sv3d-v21-nolighteven-odd_random-v4-no-dnorm_2024-12-27-14-20-44.248265.json',
        'results/zero123-v21-nolighteven-odd_random-v4-black-gt_2025-02-27-11-16-05.960585.json',
        'results/mvdfusion-v21-nolighteven-odd_random-v4_2025-03-01-16-05-51.361362.json',
        'results/sv3d-v21-elev030even-odd_random-v4-v16_2025-03-01-14-58-36.804149.json',
        'results/raw/zero123-v21-elev030even-odd_random-v4-v16_2025-03-01-14-59-49.190668.pkl',
        'results/sel10+v3d-v18-elev000-amb1.0even-odd_random-v4-white-gt_2025-03-01-20-48-27.004684.json',
        'results/raw/sel10v2+v3d-v18-elev000-amb1.0even-odd_random-v4-white-gt_2025-03-02-00-12-10.000396.pkl',
        'results/raw/sel10v3+v3d-v18-elev000-amb1.0even-odd_random-v4-white-gt_2025-03-02-11-12-55.010844.pkl',
        'results/sel10+mvdfusion-v16-elev030-amb1.0even-odd_random-v4-white-gt_2025-03-01-20-45-19.995542.json',
        'results/sel11+mvdfusion-v16-elev030-amb1.0even-odd_random-v4-white-gt_2025-03-01-21-56-03.350358.json',
        'results/raw/sel11v2+mvdfusion-v16-elev030-amb1.0even-odd_random-v4-white-gt_2025-03-01-23-21-28.364893.pkl',
        'results/sel11v3+mvdfusion-v16-elev030-amb1.0even-odd_random-v4-white-gt_2025-03-01-23-43-27.128079.json',
        'results/raw/sel11v4+mvdfusion-v16-elev030-amb1.0even-odd_random-v4-white-gt_2025-03-02-02-22-45.656010.pkl',
        'results/sel12+mvdfusion-v16-elev030-amb1.0even-odd_random-v4-white-gt_2025-03-01-21-53-11.119700.json',
        'results/syncdreamer+mvdfusion-v16-elev030-amb1.0+i000+sel11even-odd_random-v4-sel_2025-03-02-00-19-07.350370.json',
        'results/syncdreamer+mvdfusion-v16-elev030-amb1.0+i000+sel11v4even-odd_random-v4-sel_2025-03-02-01-03-08.885377.json',
        'results/cape-conv-kiui+sv3d-v21-nolight+step-050000-mergedeven-odd_random-v4_2024-12-30-13-59-54.157802.json'
    ]
    legends = [osp.basename(x)[:27] for x in pkl_files]

    # for sending out
    pkl_files = [
        'results/sv3d-v21-elev030even-odd_random-v4-v16_2025-03-01-14-58-36.804149.json',
        'results/raw/zero123-v21-elev030even-odd_random-v4-v16_2025-03-01-14-59-49.190668.pkl',
        # 'results/sel10+v3d-v18-elev000-amb1.0even-odd_random-v4-white-gt_2025-03-01-20-48-27.004684.json',
        'results/sel10v3+v3d-v18-elev000-amb1.0even-odd_random-v4-white-gt_2025-03-02-11-12-55.010844.json',
        'results/raw/sel11v2+mvdfusion-v16-elev030-amb1.0even-odd_random-v4-white-gt_2025-03-01-23-21-28.364893.pkl',
        # 'results/raw/syncdreamer+mvdfusion-v16-elev030-amb1.0+i000+sel11v2even-odd_random-v4-sel_2025-03-02-01-01-05.975610.pkl',
        # 'results/sv3dp+sv3d-v21-nolighteven-odd_random-v4-no-depth-norm_2025-01-08-00-47-13.630875.json',
        # 'results/cape-conv-kiui+sv3d-v21-nolight+step-050000-mergedeven-odd_random-v4_2024-12-30-13-59-54.157802.json'
    ]
    legends = ['GT 20 views (sv3d)', 'GT 20 views (zero123)', 'GT 18 views (v3d)', 'GT 16 views (syncdreamer)',
               'pred 16 views (syncdreamer)', 'pred 20 views sv3d', 'pred 20 views ours']
    # legends = [x[:28] for x in legends]

    # performance vs. data
    # pkl_files = [
    #     'results/cape-conv-kiui-10k+gso100-sv3d-v21-elev012-amb1.0+i000+step-050000-mergedeven-odd_random-v4_2025-03-02-00-05-44.294640.json',
    #     'results/cape-conv-kiui-50k+gso100-sv3d-v21-elev012-amb1.0+i000+step-050000-mergedeven-odd_random-v4_2025-01-06-12-00-24.739698.json',
    #     'results/cape-conv-kiui-100k+gso100-sv3d-v21-elev012-amb1.0+i000+step-050000-mergedeven-odd_random-v4_2025-01-07-05-05-20.775343.json',
    #     'results/cape-conv-kiui+gso100-sv3d-v21-elev012-amb1.0+i000+step-050000-mergedeven-odd_random-v4_2025-01-05-19-46-20.270859.json',
    #     'results/cape-conv-kiui-10k+co3d2seq-sv3d-v21-manual+i000+step-050000-mergedeven-odd_align-elev+-15_2025-03-02-00-35-09.099702.json',
    #     'results/cape-conv-kiui-50k+co3d2seq-sv3d-v21-manual+i000+step-050000-mergedeven-odd_align-elev+-15_2025-01-07-00-46-20.098586.json',
    #     'results/cape-conv-kiui-100k+co3d2seq-sv3d-v21-manual+i000+step-050000-mergedeven-odd_align-elev+-15_2025-01-07-04-51-16.750322.json',
    #     'results/cape-conv-kiui+co3d2seq-sv3d-v21-manual+i000+step-050000-mergedeven-odd_align-elev+-15_2025-01-07-17-49-21.523905.json'
    # ]

    # NeuS2 representation
    # pkl_files = [
    #     'results/sel11v4+mvdfusion-v16-elev030-amb1.0even-odd_neus2_2025-05-12-22-50-12.939527.json',
    #     'results/sel10v3+v3d-v18-elev000-amb1.0even-odd_neus2_2025-05-12-22-48-23.875156.json',
    #     # 'results/sv3d-v21-nolighteven-odd_neus2_2025-05-12-17-42-20.693189.json',
    #     'results/sv3d-v21-elev030even-odd_neus2_2025-05-13-17-54-06.943668.json',
    #     'results/zero123-v21-elev030even-odd_neus2_2025-05-14-20-14-15.789125.json'
    #     # 'results/zero123-v21-elev015even-odd_neus2_2025-05-12-22-49-03.444573.json'
    #     # 'results/zero123-v21-nolighteven-odd_neus2_2025-05-12-17-55-18.906632.json'
    # ]
    legends = ['16 views', '18 views', '20 views SV3D', '20 views zero123']

    # pat = 'align-elev+-15'
    # pkl_files = sorted(glob(f'results/*co3d2seq*_{pat}_*.json'))
    # pkl_files = sorted(glob(f'results/cape*mvimgnet*_{pat}_*.json'))
    # legends = [osp.basename(x).split('+')[0] + '+' + osp.basename(x).split('+')[1][:1] for x in pkl_files]

    # use new test views
    pat = 'random-v4'
    # pat = 'align-elev+-15'
    # pkl_files = sorted(glob(f'results/*+omni202*random-v4*.json'))
    # pkl_files = sorted(glob('results/sv3dp+co3d*seed*.json'))
    pkl_files = sorted(glob('results/*neus2-rerender*.json'))
    # pkl_files = sorted(glob(f'results/*+co3d2seq*_{pat}*.json'))
    # pkl_files = [x for x in pkl_files if 'white' not in x and '021000-' not in x and '-018750-' not in x and 'syncv2blend-slow' not in x ]
    # pkl_files = [x for x in pkl_files if ('sel1' not in x and 'v16' not in x and 'v18' not in x) or ('v16' in x and 'sel1' in x) or ('v18' in x and 'sel' in x) ]
    pkl_files = [x for x in pkl_files if 'step0' not in x or ('step0' in x and ('cape-conv-kiui+' in x or 'sv3dp-kiui+' in x))]
    # print([osp.basename(x) for x in pkl_files])
    # pat = 'random-v4'
    # pkl_files = ['results/rcn-conv-kiui+co3d2seq-sv3d-v21-manual+i000+step-050000-mergedeven-odd_align-elev+-15_2025-02-07-19-24-11.272623.json'] + sorted(glob(f'results/rcn-conv-real*+co3d*_*.json'))
    # pkl_files = sorted(glob(f'results/rcn-conv*sv3d-v21-nolight*_*.json'))
    # pkl_files = sorted(glob(f'results/stable-zero123+[g|o|c|m][s|m|o|v][o|n|3|i]*_*.json'))
    # pat = '*'
    # pkl_files = sorted(glob(f'results/cape-conv-kiui-10k+[c|m|g|o]*_*_*.json'))
    # print(pkl_files)
    # pkl_files = [x for x in pkl_files if '+' not in x]
    # legends = ['+' + osp.basename(x).split('+')[1][:18] for x in pkl_files]
    # legends = [osp.basename(x).split('+')[0][0:25] + '+' + osp.basename(x).split('+')[1][:6] + '+' + osp.basename(x).split('+')[3][:7] for x in pkl_files]
    legends = [osp.basename(x)[:28] for x in pkl_files]
    # legends = [osp.basename(x).split('+')[0][8:27] for x in pkl_files]
    # pkl_files = sorted(glob('results/*random-v4-white-gt-gso30*.json'))
    # pkl_files = [
    #     'results/zero123-v21-nolighteven-odd_random-v4_2024-12-25-19-27-49.977462.json',
    #     'results/sv3d-v21-nolighteven-odd_random-v4_2024-12-25-19-22-54.258110.json',
    #     # ''
    # ]
    # pkl_files = pkl_files[1:] + pkl_files[:1] + ['results/sv3d-v21-nolight+whiteeven-odd_white-background_2025-01-05-20-44-43.218231.json']
    # legends = [osp.basename(x).split('+')[1][7:27] for x in pkl_files]
    # legends = [osp.basename(x).split('+')[1][0:20] for x in pkl_files]


    keys = ['Chamf', 'depth',  "PSNR", 'SSIM', 'LPIPS', 'FID', 'overall', 'class', 'color', 'style']
    keys_print = ['Chamf', 'depth', "cPSNR", 'cSSIM', 'cLPIPS', 'FID', 'IQ-vlm', 'class', 'color', 'style']
    scores = {k:[] for k in keys} # for computing variance
    # keys = ['FID']
    arrows = [down_arrow,  down_arrow, up_arrow, up_arrow, down_arrow, down_arrow, up_arrow, up_arrow, up_arrow, up_arrow]
    add_std = args.add_std
    if add_std:
        print(f'{"Method":<28} |' + '|'.join([f'{(x+a).center(12)}' for x, a in zip(keys_print, arrows)]))
    else:
        print(f'{"Method":<28} |' + '|'.join([f'{(x+a).center(7)}' for x,a in zip(keys_print, arrows)]))
    # errors = [pkl.load(open(file, 'rb')) for file in pkl_files]
    normalize=False
    ofid = True
    errors_all = []
    for i, file in enumerate(pkl_files):
        if file.endswith('.json'):
            file = osp.join(osp.dirname(file), 'raw', osp.basename(file).replace('.json', '.pkl'))

        errors = pkl.load(open(file, 'rb'))
        method = osp.basename(file).split('+')[0]
        es = f'{method[:13]+legends[i]:<20} | '
        es = f'{legends[i]:<28} | '

        # rname = osp.basename(file).split('+')[1]  # get rname
        # gt_file = sorted(glob(f'results/raw/*{rname}*_random-v4-white-gt_*.pkl'))[-1]
        # print(f"{method} {gt_file}")
        # continue

        # count yes from VLM results
        name_even = errors['name_even']
        keys_vlm = ['overall', 'class', 'color', 'style']
        counts = count_vlm_results(keys_vlm, name_even)
        for key in keys_vlm:
            if counts['total'] == 0:
                errors[key] = 0
            else:
                errors[key] = counts[key] / counts['total']

        for ki, key in enumerate(keys):
            err = np.mean(errors[key])
            scores[key].append(err)

            if normalize and key in ['Chamf',  'depth', "PSNR", 'SSIM', 'LPIPS']:
                rname = osp.basename(file).split('+')[1] # get rname
                gt_file = sorted(glob(f'results/raw/*{rname}*_random-v4-white-gt_*.pkl'))[-1]
                gt_err = pkl.load(open(gt_file, 'rb'))[key]
                if key in ['PSNR', 'SSIM']:
                    err = err / np.mean(gt_err)
                elif key == 'Chamf':
                    err = 1 - (err - np.mean(gt_err))/23.
                elif key == 'LPIPS':
                    err = 1 - (err - np.mean(gt_err))/0.5
                elif key == 'depth':
                    err = 1 - (err - np.mean(gt_err))/68

            # replace dataset fid with object fid
            if ofid and key == 'FID':
                ind = osp.basename(file).find('even-odd')
                dataset = 'gso100' if '+gso100' in file else 'co3d2seq'
                if 'co3d2seq' in file:
                    fid_objects = pkl.load(open('results/fid/FID-object-clip-co3d2seq-fid-31.pkl', 'rb'))['objects']
                elif 'gso100' in file:
                    fid_objects = pkl.load(open(f'results/fid/FID-object-clip-gso100-fid-43.pkl', 'rb'))['objects']
                elif 'mvimgnet' in file:
                    fid_objects = pkl.load(open(f'results/fid/FID-object-clip-mvimgnet230-fid-27.pkl', 'rb'))['objects']
                else:
                    fid_objects = pkl.load(open(f'results/fid/FID-object-clip-omni202-34.pkl', 'rb'))  # ['objects']
                ind = ind - 8 if 'sel1' in file else ind
                # errors['FID'] = [x for x in fid_objects[osp.basename(file)[:ind]+'_even'].values()]
                if osp.basename(file)[:ind]+'_even' not in fid_objects:
                    # print('no fid for, file', file)
                    # continue
                    pass
                else:
                    err = np.mean([x for x in fid_objects[osp.basename(file)[:ind]+'_even'].values()])
                    errors[key] = [err]
            # print(osp.basename(file).split('+')[0], np.mean(errors['FID']))

            if add_std:
                es += f'{err:6.3f}{plus_minus}{np.std(errors[key]):.2f} |'
            else:
                es += f'{err:6.4f} |'
        print(es)
        errors['method'] = legends[i]
        errors_all.append(errors)
    # return
    # now print for Overleaf
    # sort based on chamf
    chamfs = [np.mean(e['Chamf']) for e in errors_all]
    indices = np.argsort(chamfs)
    # indices = range(len(chamfs))
    for ind in indices:
        ss = errors_all[ind]['method'] + ' & ' + f' & '.join([f'{np.mean(errors_all[ind][k]):.2f}' for k in keys]) + ' \\\\'
        # print(ss)
    ss = 'all avg' + ' & ' + f' & '.join([f'{np.mean(scores[k]):.4f}' for k in keys]) + ' \\\\'
    print(ss)
    ss = 'all std' + ' & ' + f' & '.join([f'{np.std(scores[k]):.4f}' for k in keys]) + ' \\\\'
    print(ss)
    ss = 'relative std ' + ' & ' + f' & '.join([f'{np.std(scores[k])/np.mean(scores[k]):.4f}' for k in keys]) + ' \\\\'
    print(ss)


def count_vlm_results(keys, name_even):
    fname = 'object_IQA-quality-1+3-new-ref'
    model_name = 'InternVL2_5-78B'
    json_files = sorted(glob(osp.join(name_even, f'*/{fname}_{model_name}.json')))
    # print(json_files)
    counts = {}
    total = 0
    for json_file in json_files:
        d = json.load(open(json_file))
        total += len(d.keys())
        for k, v in d.items():
            if k == 'prompts':
                total -= 1
                continue
            for idx, ans in enumerate(v):
                if keys[idx] not in counts:
                    counts[keys[idx]] = 0
                if 'yes' in ans.lower():
                    counts[keys[idx]] += 1
    counts['total'] = total
    # print(total, name_even)
    return counts


def print_matrix():
    "print a matrix"



def print_diff():
    "print difference between our and sv3dp fine tuned"
    pkl_files = [
        # 'results/raw/sv3dp-lvis-static+omni202-sv3d-v21-elev012-amb1.0+step-010500-mergedeven-odd_None_2024-11-19-19-52-27.152159.pkl',
        # 'results/raw/cape-conv-static-80k+omni202-sv3d-v21-elev012-amb1.0+step-078000-mergedeven-odd_None_2024-11-19-08-46-59.609110.pkl'
        'results/raw/sv3dp-lvis-static+gso100-sv3d-v21-elev012-amb1.0+step-010500-mergedeven-odd_None_2024-11-19-18-17-00.194460.pkl',
        'results/raw/cape-conv-static-80k+gso100-sv3d-v21-elev012-amb1.0+step-078000-mergedeven-odd_None_2024-11-19-08-43-47.616415.pkl'
    ]
    errors = [pkl.load(open(file, 'rb')) for file in pkl_files]
    # import pdb;pdb.set_trace()
    L = len(errors[0]['Chamf'])
    matches = []
    for i in range(L):
        cds = [x['Chamf'][i] for x in errors]
        files = [osp.basename(x['files_3d'][i][0]) for x in errors]
        psnrs = [np.mean(x['PSNR'][i]) for x in errors]
        diff_cd = cds[1] - cds[0]
        diff_psnr = psnrs[1] - psnrs[0]
        match = ((diff_cd>0) and (diff_psnr<0)) or ((diff_cd<0) and (diff_psnr>0))
        gap = diff_psnr - diff_cd
        if not match:
            print(f'{files[0][:20]:<20} | {diff_cd:6.3f} | {diff_psnr:6.3f} | {match} | {gap:.3f}')
        matches.append(match)
    print(f"Matches: {np.sum(matches)}/{L}") # Matches: 132/201 (omni202), or 61/100 (GSO100)


def print_mvpnet_manual():
    "print the statistics of mvpnet manual, based on manually annotated elevations"
    pkl_files = [
        # 'results/raw/cape-conv-kiui+mvpnet50-sv3d-v21-manual-elev030+i000+step-050000-mergedeven-odd_align-to-elev30_2024-12-09-23-51-03.140989.pkl',
        # 'results/raw/cape-conv-kiui+mvpnet50-sv3d-v21-manual+step-050000-mergedeven-odd_align-to-elev30_2024-12-09-23-53-25.393131.pkl'
        # 'results/raw/cape-conv-kiui+mvpnet50-sv3d-v21-manual-elev030+i000+step-050000-mergedeven-odd_no-align_2024-12-09-23-24-00.471878.pkl',
        # 'results/raw/cape-conv-kiui+mvpnet50-sv3d-v21-manual+step-050000-mergedeven-odd_no-align_2024-12-09-23-25-32.223300.pkl',

        'results/raw/sv3dp+mvpnet50-sv3d-v21-manual-elev030+i000even-odd_render256-align-to-elev030_2024-12-09-17-27-31.336632.pkl',
        'results/raw/sv3dp+mvpnet50-sv3d-v21-manual+i000even-odd_render256-align-to-elev030_2024-12-09-17-28-03.226159.pkl',
        # 'results/raw/sv3dp+mvpnet50-sv3d-v21-manual-v2+i000even-odd_align-icp-rot_2024-12-10-15-04-12.867224.pkl'
        # 'results/raw/sv3dp+mvpnet50-sv3d-v21-manual-elev030+i000even-odd_render256_2024-12-09-16-31-04.252424.pkl',
        # 'results/raw/sv3dp+mvpnet50-sv3d-v21-manual+i000even-odd_render256_2024-12-09-16-24-22.933986.pkl',
        # 'results/raw/syncdreamer+mvpnet50-mvdfusion-v16-manual-elev030even-odd_render256-align_2024-12-09-11-08-19.768502.pkl',
        # 'results/raw/syncdreamer+mvpnet50-mvdfusion-v16-manualeven-odd_re-render256-resample-align_2024-12-05-17-04-11.610665.pkl',

    ]
    errors = [pkl.load(open(file, 'rb')) for file in pkl_files]
    # load annotations
    ann = json.load(open('../MVD-render/splits/mvpnet50-manual.json'))
    elevs = np.array([ann[k][1] for k in sorted(ann.keys())])
    legends = ['elev30', 'manual', 'difference']
    for key in ['Chamf', 'PSNR', 'depth']:
        print(f"Metric: {key}")
        # for es, ee in zip([-60, 0, 20, 40, 60, 80], [0, 20, 40, 60, 80, 100]):
        print(f'{"interval":<11}|' + '|'.join(legends))
        print('overall:        '+ '|'.join([f'{np.mean(e[key]):.3f}' for e in errors]))
        for es, ee in zip([-60, 0, 10, 20, 30, 40, 50, 60, 80], [0, 10, 20, 30, 40, 50, 60, 80, 100]):
            mask = (elevs >= es) & (elevs < ee)
            ss = f'[{es:03d}, {ee:03d}) ({np.sum(mask):02d} objs)| '
            err_masked = []
            for err in errors:
                err_array = np.array(err[key])
                assert len(err_array) == len(elevs), f'{len(err_array)} != {len(elevs)}, {key}'
                if np.sum(mask) == 0:
                    ss += '-' * 5
                    err_masked.append(0)
                    continue
                em = np.mean(err_array[mask])
                ss += f'{em:5.3f}|'
                err_masked.append(em)
            ss += f'{err_masked[-1]-err_masked[-2]:5.3f}|' if np.sum(mask) > 0 else '--'*3
            print(ss)

def print_gso_vs_co3d():
    ""
    files_gso = sorted(glob('results/raw/*+gso100*random-v4_*'))
    files_co3d = sorted(glob('results/raw/*+co3d2seq*align-elev+-15*'))
    legends = [osp.basename(x).split('+')[0][0:13] + '+' + osp.basename(x).split('+')[1][:2] for x in files_gso]

    assert len(files_gso) == len(files_co3d)
    keys = ['Chamf', "PSNR", 'overall']
    keys_print = ['Chamf', "PSNR", 'IQ-vlm']
    arrows = [''] * len(keys)
    errors_all = []
    print(f'{"Method":<22} |' + '|'.join([f'{(x + a).center(7)}' for x, a in zip(keys_print+keys_print, arrows+arrows)]))
    for i, file in enumerate(files_gso):
        if file.endswith('.json'):
            file = osp.join(osp.dirname(file), 'raw', osp.basename(file).replace('.json', '.pkl'))

        errors = pkl.load(open(file, 'rb'))
        method1 = osp.basename(file).split('+')[0]
        method2 = osp.basename(files_co3d[i]).split('+')[0]
        assert method1 == method2, f'{method1} != {method2}'
        es = f'{legends[i]:<20} | '
        # count yes from VLM results
        name_even = errors['name_even']
        keys_vlm = ['overall', 'class', 'color', 'style']
        counts = count_vlm_results(keys_vlm, name_even)
        for key in keys_vlm:
            if counts['total'] == 0:
                errors[key] = 0
            else:
                errors[key] = counts[key] / counts['total']

        errors_other = pkl.load(open(files_co3d[i], 'rb'))
        counts = count_vlm_results(keys_vlm, errors_other['name_even'])
        for key in keys_vlm:
            if counts['total'] == 0:
                errors_other[key] = 0
            else:
                errors_other[key] = counts[key] / counts['total']

        for ki, key in enumerate(keys):
            es += f'{np.mean(errors[key]):6.4f} |'

        for ki, key in enumerate(keys):
            es += f'{np.mean(errors_other[key]):6.4f} |'
        print(es)
        # merge two dict
        errors['method'] = method1
        new_dict = {k+'_co3d':v for k,v in errors_other.items()}
        new_dict.update(**errors)
        errors_all.append(new_dict)
    #now print for overleaf
    chamfs = [np.mean(e['Chamf']) for e in errors_all]
    indices = np.argsort(chamfs)
    keys_comb = keys + [k+'_co3d' for k in keys]
    for ind in indices:
        ss = errors_all[ind]['method'] + ' & ' + f' & '.join([f'{np.mean(errors_all[ind][k]):.2f}' for k in keys_comb]) + ' \\\\'
        print(ss)



def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-as', '--add_std', default=False, action='store_true')
    args = parser.parse_args()

    print_results(args)
    # print_diff()
    # print_mvpnet_manual()
    # print_gso_vs_co3d()

if __name__ == '__main__':
    main()