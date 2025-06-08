"""
simple wrapper to run image to test
"""
import sys, os

import cv2
from tqdm import tqdm

import internvl_utils
import numpy as np
import os.path as osp
import imageio
import torch
import matplotlib.pyplot as plt
import glob
from transformers import AutoModel, AutoTokenizer
# from demo import load_image
import textwrap


class Image2Text:
    def __init__(self, args):
        "load model and initialize everything "
        path = args.path
        if args.num_gpus == 1:
            model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True).eval().cuda()
        else:
            # multi-gpu
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
        self.generation_config = dict(max_new_tokens=2048, do_sample=False, pad_token_id=tokenizer.eos_token_id) # eos for open-end generation, report bug!
        self.outdir = 'output'
        self.model_name = osp.basename(args.path)


    def test(self, args):
        model, tokenizer = self.model, self.tokenizer
        generation_config = self.generation_config

        # process the renderings
        root = '/home/ubuntu/efs/static/GSO30'
        save_name = 'GSO30-p9'

        # root = '/home/ubuntu/efs/static/co3dv2/synth-format/'
        # save_name = 'co3dv2-p8'
        model_name = osp.basename(args.path)
        save_name, root = args.save_name, args.data_path
        folders = sorted(glob.glob(osp.join(root, "*")))# + sorted(glob.glob("/home/ubuntu/efs/static/gso-50/*"))
        for folder in tqdm(folders):
            # files = sorted(glob.glob(osp.join(folder, "sv3d-v21-nolight/*.png")))[::5] # only five images
            # files = sorted(glob.glob(osp.join(folder, "sv3d-v21-frame00001/*.png")))[:1]# only the first
            files = sorted(glob.glob(osp.join(folder, f"{args.render_name}/*.png"))) + sorted(glob.glob(osp.join(folder, f"{args.render_name}/*.jpg")))
            files = files[::max(len(files)//5, 1)][:5]
            if len(files) == 0:
                print(f'no files in {folder}/{args.render_name}')
                continue
            outfile = f'output/unified/{save_name}+{osp.basename(folder)}_{model_name}.txt'
            if osp.isfile(outfile):
                continue
            outfile = f'output/unified/{save_name}+{osp.basename(folder)}_{model_name}.jpg'
            os.makedirs(osp.dirname(outfile), exist_ok=True)
            # prompt = 'here are the five multi-view renderings of an object, describe which object it is, what is the color style and textures of this object.'
            # prompt = 'here are the five multi-view renderings of an object, describe which object it is, what is the color style and textures of this object. Be concise, direct and summarize each point summarized into one sentence.'
            # prompt = ('here are the five multi-view renderings of an object, describe which object it is, the color style, geometric structure and textures of this object. '
            #     'Be concise, direct and each aspect should be summarized into one sentence.') # this result is more concise
            # prompt = ('here are the five multi-view renderings of an object, describe which object it is, the color style, geometric structure and textures of this object. '
            #           'Be concise, direct and summarize each aspect into one sentence.')  # it can indeed separate the aspects into sentences, but if too simple, it will be one sentence again
            # prompt = ('here are the five multi-view renderings of an object, describe which object it is, the color and texture style, geometric structure of this object. '
            #     'Be concise, direct and each aspects should be summarized into separate sentences.') # this will list bullet points which is not good.
            # prompt = ('here are the five multi-view renderings of an object, describe which object it is, the color and texture style, geometric structure of this object. '
            #     'Be concise, direct and summarize everything into 1-3 sentences. Always start the paragraph with "A 360 rotating view video of [the object]..."')
            prompt = (
                'here are the five multi-view renderings of an object, describe which object it is, the color and texture style, geometric structure of this object. '
                'Be concise, direct and summarize everything into 1-3 sentences. Always start the paragraph with "A 360 rotating view video of [the object]..."')
            pixel_values = [internvl_utils.load_image(x, max_num=12, resize_width=(448, 448)).to(torch.bfloat16).cuda() for x in files] # this is 10,3,448,448 if do not resize to 512
            # print(files)

            # images for visualization
            image = np.concatenate([cv2.resize(cv2.imread(x)[:, :, ::-1], (512, 512)) for x in files], 1)
            # import pdb;pdb.set_trace()
            # num_patches_list = [x.size(0) for x in pixel_values]
            pixel_input = torch.cat(pixel_values, dim=0)
            self.prompt_and_save(generation_config, image, model, outfile, pixel_input, prompt, tokenizer)


    def prompt_and_save(self, generation_config, image, model, outfile, pixel_input, prompt, tokenizer,
                        add_prompt=False):
        response, history = model.chat(tokenizer, pixel_input, prompt, generation_config,
                                       # num_patches_list=num_patches_list,
                                       history=None,
                                       return_history=True)  # for combined image, no need to use num_patches_list
        # now combine text and image
        response = self.visualize_response(add_prompt, image, outfile, prompt, response)
        # also save the text
        with open(outfile.replace('.jpg', '.txt'), 'w') as f:
            f.write(prompt + '\n')
            f.write(response)

        # print('Image saved to', outfile)

    def visualize_response(self, add_prompt, image, outfile, prompt, response, text_width=145,
                           font_size=5):
        fig, ax = plt.subplots()
        ax.imshow(image)
        if add_prompt:
            response = "Prompt:" + prompt + '\n' + response  # also add prompt to the annotation
        wrapped_text = textwrap.fill(response, width=text_width)  # Adjust the width as needed
        ax.text(10, 10, wrapped_text, color='white', fontsize=font_size, bbox=dict(facecolor='black', alpha=0.7))
        # Remove axis labels
        ax.axis('off')
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        return response


class Image2TextMultiRound(Image2Text):
    "multiple rounds to describe color, appearance style, object class"
    def prompt_and_save(self, generation_config, image, model, outfile, pixel_input, prompt, tokenizer,
                        add_prompt=False):
        ""
        # apperance style
        prompt_app = ("Here are images of a daily object, what is the appearance style of this object? Ignore the background, focus on the appearance, style and design instead of describing the object type, "
                      "return the appearance style only and in less than 5 words.")
        appearance, history = model.chat(tokenizer, pixel_input,
                                         prompt_app,
                                         generation_config,
                                         history=None,
                                         return_history=True)

        prompt_class = "Which object it is? Just return the class name, do not repeat question. Use daily used common words. If there are multiple possibilities, return like this: classname 1 or classname2 or classname3..."
        # ask about class
        obj_class, history = model.chat(tokenizer, pixel_input, prompt_class, generation_config,
                                       history=history,
                                       return_history=True)

        # color
        prompt_color = "What is the main color(s) of this object? simply answer the color(s), summarize to less than 4 colors."
        color, history = model.chat(tokenizer, pixel_input,
                                        prompt_color,
                                        generation_config,
                                        history=history,
                                        return_history=True)
        prompt = ('what is the object shown in the images? describe the object type, color, appearance style and geometric structure of the object. '
                 'Be concise and summarize everything into 1-3 sentences. Start the sentence directly with the object class name.')
        # comprehensive
        paragraph, history = model.chat(tokenizer, pixel_input,
                                    prompt,
                                    generation_config,
                                    history=None,
                                    return_history=True)

        # do batch chat, this requires more memory, but faster
        # prompts = [prompt_class, prompt_app, prompt_color, prompt]
        # pixels = [pixel_input] * len(prompts)
        # num_patches_list = [x.size(0) for x in pixels]
        # pixels = torch.cat(pixels, dim=0)
        # # pixels = torch.cat(pixels, dim=0)
        # responses = model.batch_chat(tokenizer, pixels,
        #                              num_patches_list=num_patches_list,
        #                              questions=prompts,
        #                              generation_config=generation_config)
        # obj_class, appearance, color, paragraph = responses

        text_comb = f'object class: {obj_class};\tapperance style: {appearance}\tcolor: {color};\t\nFull text: {paragraph}'
        # now combine text and image
        fig, ax = plt.subplots()
        ax.imshow(image)
        if add_prompt:
            response = "Prompt:" + prompt + '\n' + text_comb  # also add prompt to the annotation
        else:
            response = text_comb
        wrapped_text = textwrap.fill(response, width=140)  # Adjust the width as needed
        ax.text(10, 10, wrapped_text, color='white', fontsize=5, bbox=dict(facecolor='black', alpha=0.7))
        # Remove axis labels
        ax.axis('off')
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        # also save the text
        with open(outfile.replace('.jpg', '.txt'), 'w') as f:
            f.write(prompt + '\n')
            f.write(paragraph + '\n')
            f.write(obj_class + '\n')
            f.write(color + '\n')
            f.write(appearance + '\n')

        print('Image saved to', outfile)


class Image2TextPerFrame(Image2Text):
    "prompt the model for each image respectively"
    def test(self, args):
        ""
        model, tokenizer = self.model, self.tokenizer
        generation_config = self.generation_config
        import pdb;pdb.set_trace()

        # root = '/home/ubuntu/efs/static/GSO30'
        root = '/home/ubuntu/efs/work/generative-models/outputs/simple_video_sample/sv3d_u-gso30-elev15/'
        save_name = 'GSO30-p9-sv3du'
        model_name = osp.basename(args.path)
        folders = sorted(glob.glob(osp.join(root, "*/")))  # + sorted(glob.glob("/home/ubuntu/efs/static/gso-50/*"))
        for folder in tqdm(folders):
            # files = sorted(glob.glob(osp.join(folder, "render/*.png")))# use all views
            files = sorted(glob.glob(osp.join(folder, "*.png")))
            folder = folder[:-1]
            if len(files) == 0:
                continue

            # test each image individually
            prompt = ('what is the objectt shown in the image? describe the object type, color, and appearance style of the object. '
                      'Be concise and summarize everything into 1-3 sentences.')
            pixel_values = [internvl_utils.load_image(x, max_num=12, resize_width=(512, 512)).to(torch.bfloat16).cuda() for x in files]  # this is 10,3,448,448 if do not resize to 512
            images = [cv2.resize(cv2.imread(x)[:, :, ::-1], (512, 512)) for x in files]
            outfolder = f'output/{save_name}/{osp.basename(folder)}/'
            os.makedirs(outfolder, exist_ok=True)
            for i, (pixel, image) in enumerate(zip(pixel_values, images)):
                # prompt and get result
                response, history = model.chat(tokenizer, pixel, prompt, generation_config,
                                               # num_patches_list=num_patches_list,
                                               history=None,
                                               return_history=True)
                # now combine text and image
                outfile = f'{outfolder}/{osp.basename(folder)}_{model_name}_{i:03d}.jpg'
                response = self.visualize_response(False, image, outfile, prompt, response, text_width=100)
                # save the text as well
                with open(outfile.replace('.jpg', '.txt'), 'w') as f:
                    f.write(prompt + '\n')
                    f.write(response)

            # make a video
            image_files = sorted(glob.glob(outfolder+"/*.jpg"))
            size = None
            video_file = osp.join(outfolder, f'{osp.basename(folder)}_{model_name}.mp4')
            vw = imageio.get_writer(video_file, 'FFMPEG', fps=3)
            for file in image_files:
                img = cv2.imread(file)
                if size is None:
                    size = img.shape[:2][::-1]
                img = cv2.resize(img, size)
                vw.append_data(img[:, :, ::-1])
            print('video saved to', video_file)
            vw.close()


class Image2TextOrigRender(Image2Text):
    "save "



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', default='pretrained/InternVL2_5-78B') # efs is faster
    parser.add_argument('-d', '--data_path')
    parser.add_argument('-ng', '--num_gpus', default=4, type=int)
    parser.add_argument('-rn', '--render_name')
    parser.add_argument('--save_name')

    args = parser.parse_args()
    # annotate real images: python images2text.py -d /home/ubuntu/efs/static/mvimgnet230 -rn crops --save_name mvimgnet230

    # options: , 'pretrained/InternVL2-26B', 'pretrained/InternVL2-8B'
    # for path in ['pretrained/InternVL2-Llama3-76B', 'pretrained/InternVL2-40B', 'pretrained/InternVL2-26B', 'pretrained/InternVL2-8B']:
    # for path in ['pretrained/InternVL2-Llama3-76B']:
    #     torch.cuda.empty_cache()
    #     os.system('nvidia-smi')
    #     args.path = path
    #     # model = Image2Text(args)
    #     model = Image2TextMultiRound(args)
    #     # model = Image2TextPerFrame(args)
    #     model.test(args)
    #     del model # this makes sure the cuda memory can be released

    model = Image2TextMultiRound(args)
    # model = Image2TextPerFrame(args)
    model.test(args)

