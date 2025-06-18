'''
python test.py --weight_path="./checkpoints/cn_d25ofd18_epoch-v18.pth" \\
               --image_path="./test/4decce85-c6ede74e-7a8bc81c-e81edee9-5ec17116.jpg" \\
               --text_prompt="Large right-sided pneumothorax." --num_samples=4
'''

import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import os
from datetime import datetime
from LungDetection.main import lungsegment
import argparse

import torchvision.transforms as T
from sentence_transformers import util
from groundingdino.util.inference import load_image
from PIL import Image
import pandas as pd

from torch.nn import CosineSimilarity
cos = CosineSimilarity(dim=0)


def get_args_parser():
    parser = argparse.ArgumentParser('Set Visual Grounding', add_help=False)
    parser.add_argument('--weight_path', type=str, default="./checkpoints/cn_d25ofd18_epoch-v18.pth", 
                        help="The path to the trained model")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--image_path', type=str, 
                        help="The path to the image file.")
    parser.add_argument('--text_prompt', type=str, 
                        help="The text prompt.")
    parser.add_argument('--num_samples', type=int, default=4, help="Number of generated samples.")
    parser.add_argument('--plot_gen_image', action='store_true')
    parser.add_argument('--output_path', type=str, default="./test/samples/output/",
                        help="The path to the generated files.")
    return parser


apply_uniformer = UniformerDetector()
apply_canny = CannyDetector()

def process(input_image, prompt, model, num_samples, image_resolution=512, ddim_steps=10, guess_mode=False, strength=1, scale=9, seed=-1, eta=0):
    with torch.no_grad():
        ddim_sampler = DDIMSampler(model)
        img = resize_image(HWC3(input_image), image_resolution)
        # detected_map = apply_uniformer(resize_image(input_image, image_resolution))
        H, W, C = img.shape

        detected_map = apply_canny(img, 100, 200)
        detected_map = HWC3(detected_map)
        # detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        # control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.from_numpy(detected_map.copy()).float().cpu() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
        #cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        #un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results

def imageEncoder(img):
    image_source, image = load_image(img)
    return image
def generateScore(image1, image2):
    # test_img = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    # data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(image1)
    img2 = imageEncoder(image2)
    score = cos(img1, img2)
    return score

def main(args):
    model = create_model('./models/cldm_v15_biovlp.yaml').cpu()
    model.load_state_dict(load_state_dict(args.weight_path, location=args.device))
    if args.device == 'cuda':
        model = model.cuda()
    

    prompt = args.text_prompt
    img_org = cv2.imread(args.image_path)
    img_w, img_h, c = img_org.shape
    input_img = lungsegment(args.image_path)
    gen_img = process(input_img, prompt, model, args.num_samples)

    if args.plot_gen_image:
        for i in range(1,len(gen_img)):
            cv2.imshow(f'sample_{i}', gen_img[i])
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    info_dict = {"gen_sample_path":[], "similarity_rate":[]}
    # current_time = datetime.now()
    # epoch = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.output_path, exist_ok=True)
    for i in range(1,len(gen_img)):
        resized = cv2.resize(gen_img[i], (img_h, img_w), interpolation = cv2.INTER_LINEAR)
        # fn = f'./test/samples/pt{epoch}/gen_out_inv_sample{i}.jpg'
        fn = args.output_path + f'gen_out_inv_sample{i}.jpg'
        cv2.imwrite(fn, resized)
        info_dict["gen_sample_path"].append(fn)
        info_dict["similarity_rate"].append(generateScore(args.image_path, fn).mean())
    with open(args.output_path+"prompt.txt", "w") as file:
        file.write(prompt)
    
    df = pd.DataFrame(info_dict)
    df.to_csv(args.output_path+"info_path_similarity.csv", index=False)

    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generating CXR Image using Prompt and conditioning with Binary image', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)