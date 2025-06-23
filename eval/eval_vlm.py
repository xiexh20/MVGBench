"""
evaluate generated images with VLM
"""
import sys, os
import cv2
import torch
sys.path.append(os.getcwd())
import os.path as osp
from copy import deepcopy
import internvl_utils, json
import numpy as np
from glob import glob
import torch
import glob
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

class VLMEvaluator:
    def __init__(self, args):
        "initialize the model"
        # multi-gpu
        path = args.path
        device_map = internvl_utils.split_model(osp.basename(args.path))
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = dict(max_new_tokens=2048, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)  # eos for open-end generation, report bug!
        self.model_name = osp.basename(args.path)
        self.out_fname = 'object_IQA-quality-1+3-new-ref'
        self.keys_vlm = ['IQ-vlm', 'class', 'color', 'style'] # key names of VLM metrics

    def evaluate(self, args):
        "run VLM and accumulate the results"
        model, tokenizer = self.model, self.tokenizer
        generation_config = self.generation_config
        prompts = [
            # IQ-vlm
            "Is this image an overall high-quality image with good overall structure, good visual quality, nice color harmony, clear object and free of strange artifacts and distortions? just answer yes or no.",
        ]

        subfolders = sorted(glob.glob(args.name_even + "/*/"))
        for folder in tqdm(subfolders, desc=osp.basename(args.name_even)):
            outfile = folder + f'/object_{self.out_fname}_{self.model_name}.json'
            if osp.isfile(outfile) and not args.redo:
                continue
            files = sorted(glob.glob(folder + "/train/ours_10000/gt/*.png")) # Directly query the MVG model generated images
            if len(files) == 0:
                print(f"Warning: no files found in {folder}/train/ours_10000/gt/!")
                continue
            semantics = self.load_semantics(folder, args.vlm_ann_path)

            obj_cls, color, style = semantics
            prompts_sem = [
                f'Is {obj_cls} presented in this image? just answer yes or no.',
                f'Does the object (possibly {obj_cls}) shown in this image have the color(s): {color}? just answer yes or no.',
                f'Is the appearance style of the object (possibly {obj_cls}): {style}? just answer yes or no.',
            ]
            pixel_values = [internvl_utils.load_image(x, max_num=12, resize_width=(448, 448)).to(torch.bfloat16).cuda()
                            for x in files]

            res_dict, prompt_dict = {}, {}  # results for this object
            for i, (file, pix) in enumerate(zip(files, pixel_values)):
                prompts_i = deepcopy(prompts_sem)
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

                # visualize
                # img = cv2.resize(np.array(Image.open(file)), (256, 256))
                # comb = []
                # for res in responses:
                #     vis = img.copy()
                #     cv2.putText(vis, res, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                #     comb.append(vis)
            res_dict['prompts'] = prompt_dict
            json.dump(res_dict, open(outfile, 'w'), indent=2)

        # Now accumulate results
        counts = self.count_vlm_results(self.keys_vlm, args.name_even)
        vlm_scores = {}
        for key in self.keys_vlm:
            if counts['total'] == 0:
                vlm_scores[key] = 0
            else:
                vlm_scores[key] = counts[key] / counts['total']
        es = ''
        for k, v in vlm_scores.items():
            es += f'{k}: {v:.2f} '
        print(es)

    def count_vlm_results(self, keys, name_even):
        json_files = sorted(glob(osp.join(name_even, f'*/{self.out_fname}_{self.model_name}.json')))
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
        return counts

    def load_semantics(self, folder, vlm_ann_path):
        # load vlm-annotation
        dname = 'gso30'  # by default dataset name
        if 'mvpnet50' in folder:
            dname = 'mvpnet50'
        elif 'gso100' in folder:
            dname = 'gso100'
        elif 'omni202' in folder:
            dname = 'omni202'
        elif 'co3d2seq' in folder:
            dname = 'co3d2seq'
        elif 'mvimgnet' in folder:
            dname = 'mvimgnet230'
        obj_name = osp.basename(folder[:-1])
        txt_file = osp.join(vlm_ann_path, f'{dname}+{obj_name}_InternVL2_5-78B.txt')
        lines = [x.replace("\n", '') for x in open(txt_file, 'r', encoding='utf-8').readlines()]
        semantics = lines[-3:]
        return semantics



if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-ne', '--name_even')
    parser.add_argument('--vlm_ann_path', default='example/vlm-ann')
    parser.add_argument('-p', '--path', default='pretrained/InternVL2_5-78B')  # efs is faster

    args = parser.parse_args()

    evaluator = VLMEvaluator(args)
    evaluator.evaluate(args)