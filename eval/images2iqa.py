"""
ask the model binary questionL simply yes and no
reference: https://arxiv.org/pdf/2410.16892
"""
import sys, os
import cv2
import torch
sys.path.append(os.getcwd())
import os.path as osp
from copy import deepcopy
import internvl_utils, json
import numpy as np
import imageio
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
import glob
from transformers import AutoModel, AutoTokenizer
import textwrap
from tqdm import tqdm

from images2text import Image2Text


class Images2IQABinary(Image2Text):
    def test(self, args):
        "given a root folder, process all images inside, e.g. root/obj/train_*.png"
        model, tokenizer = self.model, self.tokenizer
        generation_config = self.generation_config
        view10_texts = ['front side', 'front side', 'side', 'back side', 'back side',
                        'back', 'back side', 'back side', 'side', 'front side']
        view8_texts = ['front side', 'front side', 'side', 'back side', 'back side',
                       'back', 'back side', 'front side']

        prompts = [
            # 'Is the image free of noise or distortion? just answer yes or no.', # this is meaningful
            # 'Does the image show clear objects and sharp edges? just answer yes or no.', # reasonable
            # 'Does this image show detailed textures and materials? just answer yes or no.', # not good
            # 'Is this image overall a high-quality image with clear objects, sharp edges, nice color, good overall structure, and good visual quality? just answer yes or no.',
            ] # not at all, overall is very bad for all cases

        # prompts = [
        #     'Is the image free of noise or distortion? just answer yes or no.',
        #     'Does the image show clear structure of the objects? just answer yes or no.',
        #     'Does the image show clean texture of the object? just answer yes or no.',
        #     "Are you able to recognize the object class with high certainty? just answer yes or no.",
        #     "Does the image show clear objects? just answer yes or no.",
        #     'Do the objects in the image look natural overall? just answer yes or no.',
        #     'Does the image look natural overall? just answer yes or no.',
        #     'Does the image look good and the objects look like actual objects overall? just answer yes or no.',
        # ]
        prompts = [
            'Is the image free of noise or distortion? just answer yes or no.',
            'Does the image show clean texture of the object? just answer yes or no.',
            "Are you able to recognize the object class with high certainty? just answer yes or no.",
            "Do the image and object have overall natural appearance? just answer yes or no.",
        ]
        fname = 'IQA-4+3-all'

        prompts = [
            # this overall quality for internvl2.5 works
            "Is this image an overall high-quality image with good overall structure, good visual quality, nice color harmony, clear object and free of strange artifacts and distortions? just answer yes or no.",
            # 'Is this image free of contents that might lead to uncanny effects? just answer yes or no.', # this is not useful!
        ]
        fname = 'IQA-quality-1+3-new-ref'
        skip = 1
        # fname = 'IQA-quality-half'
        # skip = 2

        subfolders = sorted(glob.glob(args.root + "/*/"))
        video_file = f'output/videos-debug/{osp.basename(args.root)}.mp4'
        video_writer = imageio.get_writer(video_file, "FFMPEG", fps=3)
        for folder in tqdm(subfolders, desc=osp.basename(args.root)):
            # outfile = folder + f'/object_IQA-all_{self.model_name}.json' # all is for all views, all questions
            # outfile = folder + f'/object_IQA-sem-set2_{self.model_name}.json'
            outfile = folder + f'/object_{fname}_{self.model_name}.json'
            if osp.isfile(outfile) and not args.redo:
                d = json.load(open(outfile))
                done = True
                continue
            files = sorted(glob.glob(folder + "/train*.png"))
            view_texts = view10_texts if len(view10_texts) == 10 else view8_texts
            view_indices = np.arange(len(files))
            files, view_indices = files[::skip], view_indices[::skip]
            # files = sorted(glob.glob(folder + "/train/ours_10000/gt/*.png")) # use 3dgs format
            if len(files) == 0:
                # print(f"Warning: no files found in {folder}!")
                files = sorted(glob.glob(folder + "/train/ours_10000/gt/*.png"))
                if len(files) == 0:
                    print(f"Warning: no files found in {folder}/train/ours_10000/gt/!")
                    continue
            name = osp.basename(folder[:-1])
            # obj_cls = self.load_internvl_text(name, 'class2')
            # obj_cls = name if '-frame00001' in folder else self.load_internvl_text(name, 'class2')
            # semantics = [self.load_internvl_text(name, x) for x in ['class2', 'class', 'full', 'color+class', 'style+class']]
            # prompts_sem = [f'Is {x} presented in this image? just answer yes or no.' for x in semantics]
            # sentence = self.load_internvl_text(name, "sentence")
            # prompts_sem.append(f'Here is description of an object: {sentence},'
            #                    f' does this image shown such an object? just answer yes or no.')
            # more complex: given the camera pose information
            # prompts_sem.append(f'Here is description of an object: {self.load_internvl_text(name, "sentence")},'
            #                    f' does this image shown such an object? just answer yes or no.')

            # IQA-sem-set2: new set of prompts
            dname = 'gso30' # by default dataset name
            if 'mvpnet50' in folder:
                dname = 'mvpnet50'
            elif 'co3dv2' in folder:
                dname = 'co3dv2'
            elif 'gso100' in folder:
                dname = 'gso100'
            elif 'omni202' in folder:
                dname = 'omni202'
            elif 'co3d2seq' in folder:
                dname = 'co3d2seq'
            elif 'mvimgnet' in folder:
                dname = 'mvimgnet230'

            semantics = [self.load_internvl_text(f'{dname}+{name}', x) for x in ['class2', 'color-only', 'style-only']]
            obj_cls, color, style = semantics
            prompts_sem = [
                f'Is {obj_cls} presented in this image? just answer yes or no.',
                f'Does the object (possibly {obj_cls}) shown in this image have the color(s): {color}? just answer yes or no.',
                # f'Is a {color} {obj_cls} presented in this image? just answer yes or no.',
                # f'Is the main color of the object {color}? just answer yes or no.',
                f'Is the appearance style of the object (possibly {obj_cls}): {style}? just answer yes or no.',
            ]

            pixel_values = [internvl_utils.load_image(x, max_num=12, resize_width=(448, 448)).to(torch.bfloat16).cuda() for x in files]

            # now ask the question for every single image
            res_dict, prompt_dict = {}, {} # 12s/obj with 5 images
            for i, (file, pix) in enumerate(zip(files, pixel_values)):
                prompts_i = deepcopy(prompts_sem)
                # view_promt = (f'Here is description of an object: {sentence}, '
                #               f'this is the image of an object in {view_texts[view_indices[i]]} view, '
                #               f'does this image show such an object in description? Note that in back views some colors can be different but the object class should be the same. '
                #               f'just answer yes or no.')
                # prompts_i.append(view_promt)
                pixs = [pix] * (len(prompts) + len(prompts_i))
                num_patches_list = [x.size(0) for x in pixs]
                pixs = torch.cat(pixs, dim=0)
                questions = deepcopy(prompts) + prompts_i
                responses = model.batch_chat(tokenizer, pixs,
                                             num_patches_list=num_patches_list,
                                             questions=questions,
                                             generation_config=generation_config)
                # save results to file
                res_dict[osp.basename(file)] = responses
                prompt_dict[osp.basename(file)] = prompts_i
                # print(osp.basename(file), obj_cls, responses)

                # visualize
                img = cv2.resize(np.array(Image.open(file)), (256, 256))
                comb = []
                for res in responses:
                    vis = img.copy()
                    cv2.putText(vis, res, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    comb.append(vis)
                video_writer.append_data(np.concatenate(comb, axis=0))
            res_dict['prompts'] = prompt_dict
            json.dump(res_dict, open(outfile, 'w'), indent=2)
        print("Visualization saved to", video_file)

    def load_internvl_text(self, name, text_type='full'):
        "load text information from InternVL model given the object name"
        # internvl_path = '/home/ubuntu/efs/work/InternVL/output/GSO30-p8/'
        # txt_file = osp.join(internvl_path, f'{name}_InternVL2-Llama3-76B.txt')
        # if not osp.isfile(txt_file):
        #     txt_file = f'/home/ubuntu/efs/work/InternVL/output/co3dv2-p8/{name}_InternVL2-Llama3-76B.txt'
        # if not osp.isfile(txt_file):
        #     txt_file = f'/home/ubuntu/efs/work/InternVL/output/mvpnet50/{name}_InternVL2_5-78B.txt'
        # if not osp.isfile(txt_file):
        #     txt_file = f'/home/ubuntu/efs/work/InternVL/output/gso100/{name}_InternVL2_5-78B.txt'
        # if not osp.isfile(txt_file):
        #     txt_file = f'/home/ubuntu/efs/work/InternVL/output/omni202/{name}_InternVL2_5-78B.txt'
        internvl_path = '/home/ubuntu/efs/work/InternVL/output/unified/'
        txt_files = sorted(glob.glob(f'{internvl_path}/{name}_InternVL2_5-78B.txt'))
        assert len(txt_files) == 1, f'more than one txt files ({len(txt_files)}) found for {name}!'
        txt_file = txt_files[0]
        lines = [x.replace("\n", '') for x in open(txt_file,'r',  encoding='utf-8').readlines()]
        obj_type, color, style = lines[-3:]
        if text_type == 'full':
            text = f'a {color} {style} {obj_type}'
        elif text_type == 'class':
            text = f'a {obj_type}'
        elif text_type == 'color+class':
            text = f'a {color} {obj_type}'
        elif text_type == 'style+class':
            text = f'a {style} {obj_type}'
        elif text_type == 'class2':
            text = obj_type
        elif text_type == 'sentence':
            txt_file = f'/home/ubuntu/efs/work/InternVL/output/GSO30-p6/{name}_InternVL2-Llama3-76B.txt'
            text = open(txt_file,'r',  encoding='utf-8').readlines()[-1].replace('\n', '')
        elif text_type == 'color-only':
            return color
        elif text_type == 'style-only':
            return style
        else:
            raise NotImplementedError
        return text


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', default='pretrained/InternVL2_5-78B') # 40B is the best among them
    parser.add_argument('-ng', '--num_gpus', default=4, type=int)
    parser.add_argument('-ro', '--root')
    parser.add_argument('--redo', default=False, action='store_true',)
    parser.add_argument('-n', '--name')

    args = parser.parse_args()

    # for path in ['pretrained/InternVL2-Llama3-76B', 'pretrained/InternVL2-40B', 'pretrained/InternVL2-26B', 'pretrained/InternVL2-8B']:

    roots = [
        # '/home/ubuntu/efs/work/SyncDreamer/output/3dgs-input/syncdreamer+mvdfusion-v16-frame00001_even',
        # '/home/ubuntu/efs/work/EscherNet/logs_6DoF/NeRF/3dgs/eschernet+zero123-v21-frame00001+i000_even',
        # '/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/3dgs-format/sv3dp+sv3d-v21-frame00001+i000_even',
        # '/home/ubuntu/efs/static/GSO30-3dgs-format/sv3d-v21-nolight_even'
        # '/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/3dgs-format/cape-lvis-cosine-lr+sv3d-v21-nolight+step-009900_even'
        '/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/3dgs-format/cape-lvis-cosine-lr+sv3d-v21-frame00001+step-009900_even'
    ]

    res_folders = [
        '/home/ubuntu/efs/work/SyncDreamer/output/3dgs-input/',
        '/home/ubuntu/efs/work/EscherNet/logs_6DoF/NeRF/3dgs/',
        '/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/3dgs-format/',
        '/home/ubuntu/efs/static/GSO30-3dgs-format/',
        '/home/ubuntu/efs/work/LGM/output/3dgs/'
    ]
    # res_folders = sorted(glob.glob(args.root)) + res_folders
    # for folder in res_folders:
    #     roots = sorted(glob.glob(folder + f"/*{args.name}"))
    #     if len(roots) == 0:
    #         print('no folders found in', args.root + args.name)
    #         exit(0)

    # semantic set 2:
    roots_gso = [
        "/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/3dgs-format/sv3dp+sv3d-v21-nolight_even",
        '/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/3dgs-format/cape-lvis-cosine-lr+sv3d-v21-nolight+step-009900_even',
        '/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/3dgs-format/cape-lvis-bs64+sv3d-v21-nolight+step-010000-merged_even',
        '/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/3dgs-format/cape-lvis-dyna+sv3d-v21-nolight+step-010000-merged_even',
        '/home/ubuntu/efs/work/EscherNet/logs_6DoF/NeRF/3dgs/eschernet+zero123-v21-nolight_even',
        '/home/ubuntu/efs/work/zero123/zero123/output/3dgs/zero123+zero123-v21-nolight_even',
        '/home/ubuntu/efs/work/SyncDreamer/output/3dgs-input/syncdreamer+mvdfusion-v16-elev030-amb1.0+i000_even',
        '/home/ubuntu/efs/work/mvdfusion/demo/3dgs-input/mvdfusion+mvdfusion-v16-elev030-amb1.0_even',
        '/home/ubuntu/efs/work/LGM/output/3dgs/imagedream+imagedream-v21-nolight_even',
        '/home/ubuntu/efs/static/GSO30-3dgs-format/sv3d-v21-nolight_even',
    ]
    roots_real = [
        '/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/3dgs-format/sv3dp+sv3d-v21-frame00001+i000_even',
        '/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/3dgs-format/cape-lvis-cosine-lr+sv3d-v21-frame00001+step-009900_even',
        '/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/3dgs-format/cape-lvis-bs64+sv3d-v21-frame00001+step-010000-merged_even',
        '/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/3dgs-format/cape-lvis-dyna+sv3d-v21-frame00001+step-010000-merged_even',
        '/home/ubuntu/efs/work/LGM/output/3dgs/imagedream+imagedream-v21-frame00001+i000_even',
        '/home/ubuntu/efs/work/SyncDreamer/output/3dgs-input/syncdreamer+mvdfusion-v16-frame00001_even',
        '/home/ubuntu/efs/work/EscherNet/logs_6DoF/NeRF/3dgs/eschernet+zero123-v21-frame00001+i000_even',
        '/home/ubuntu/efs/work/mvdfusion/demo/3dgs-input/mvdfusion+mvdfusion-v16-frame00001_even',
        '/home/ubuntu/efs/work/zero123/zero123/output/3dgs/zero123+zero123-v21-frame00001+i000_even'
        # '/home/ubuntu/efs/static/GSO30-3dgs-format/sv3d-v21-nolight_even'
        # MVDFusion: run on L40x4small-26, zero123: to be started.
    ]
    # roots_cape = [
    #     # '/home/ubuntu/efs/work/generative-models/outputs/sample/3dgs-format/cape-conv-static-80k+sv3d-v21-nolight+step-009000-merged_even',
    #     # '/home/ubuntu/efs/work/generative-models/outputs/sample/3dgs-format/cape-conv-static-80k+sv3d-v21-nolight+step-008500-merged_even',
    #     # '/home/ubuntu/efs/work/generative-models/outputs/sample/3dgs-format/cape-conv-static-80k+sv3d-v21-nolight+step-007000-merged_even',
    #     # '/home/ubuntu/efs/work/generative-models/outputs/sample/3dgs-format/cape-conv-static-80k+sv3d-v21-nolight+step-006000-merged_even',
    #     # '/home/ubuntu/efs/work/generative-models/outputs/sample/3dgs-format/cape-conv-static-80k+sv3d-v21-nolight+step-005000-merged_even'
    #     '/home/ubuntu/efs/work/generative-models/outputs/sample/3dgs-format/cape-conv-static-80k+sv3d-v21-nolight+step-010000-merged_even',
    #     '/home/ubuntu/efs/work/generative-models/outputs/sample/3dgs-format/cape-conv-static-80k+sv3d-v21-frame00001+step-010000-merged_even'
    # ]
    # roots_cape = roots_cape + roots_gso

    # GSO30 objects
    names = [
        'output/consistency/cape-conv-kiui+sv3d-v21-nolight+step-050000-merged_even',
        'output/consistency/sv3dp-lvis-static+sv3d-v21-nolight+step-017000-merged_even',
        'output/consistency/epidiff+mvdfusion-v16-nolight+i000_even',
        'output/consistency/eschernet+zero123-v21-nolight_even',
        'output/consistency/free3d+zero123-v21-nolight+i000_even',
        'output/consistency/sv3dp+sv3d-v21-nolight_even',
        'output/consistency/syncdreamer+mvdfusion-v21-nolight_even',
        'output/consistency/zero123+zero123-v21-nolight+i000_even',
        'output/consistency/mvdfusion+mvdfusion-v16-nolight_even',
        'output/consistency/imagedream-v21-nolight_even',
        '/home/ubuntu/efs/work/LGM/output/3dgs/imagedream+imagedream-v21-nolight_even',
    ]
    # roots = [osp.join('/home/ubuntu/efs/work/gaussian-splatting/', x) for x in names]
    # roots = sorted(glob.glob('/home/ubuntu/efs/work/gaussian-splatting/output/consistency/*+mvpnet50*-manual-v2*_even'))
    roots = sorted(glob.glob(args.root))
    roots = [x for x in roots if 'step-0195' not in x and 'step-0445' not in x and 'step-0105' not in x and 'step-078' not in x]
    print([osp.basename(x) for x in roots], len(roots))

    runner = Images2IQABinary(args)
    for root in tqdm(roots):
        args.root = root
        try:
            runner.test(args)
        except Exception as e:
            import traceback
            print(e)
            print(traceback.format_exc())


    # runner = Images2IQABinary(args) # initialize only once, otherwise oom!
    # os.system('nvidia-smi')
    # for name in ['even', 'odd']:
    #     for folder in res_folders:
    #         for pat in ['nolight', 'frame000']:
    #             roots = sorted(glob.glob(folder + f"/*{pat}*_{name}"))
    #             # print(roots)
    #             # continue
    #
    #             print(f"In total {len(roots)} folders in {folder}", roots)
    #             for root in roots:
    #                 skip = False
    #                 # for k in ['omni3d', ]
    #                 if 'omni3d' in root or 'frame000' in root:
    #                     continue
    #                 args.root = root
    #                 try:
    #                     runner.test(args)
    #                 except Exception as e:
    #                     import traceback
    #
    #                     print(e)
    #                     print(traceback.format_exc())

    # roots = sorted(glob.glob("/home/ubuntu/efs/work/gaussian-splatting/output/consistency/*nolight*_even"))
    # for folder in res_folders:
    #     roots = sorted(glob.glob(folder + f"/*{args.name}"))
        # if len(roots) == 0:
        #     print('no folders found in', args.root + args.name)
        #     exit(0)
        # print(f"In total {len(roots)} folders in {folder}", roots)
        # for root in roots:
        #     if 'omni3d' in root or 'frame000' in root:
        #         continue
        #     args.root = root
        #     try:
        #         runner.test(args)
        #     except Exception as e:
        #         import traceback
        #         print(e)
        #         print(traceback.format_exc())