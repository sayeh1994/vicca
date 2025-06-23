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
import pickle
from PIL import Image
from term_image.image import AutoImage

apply_uniformer = UniformerDetector()
apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
# model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cuda'))
model.load_state_dict(load_state_dict('./checkpoints/cn_d25ofd18_epoch-v18.pth', location='cuda'))
#model.load_state_dict(load_state_dict('./checkpoints/cn_d12_epoch-v17.pth'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, num_samples=10, image_resolution=512, ddim_steps=20, guess_mode=False, strength=1, scale=9, seed=-1, eta=0):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        # detected_map = apply_uniformer(resize_image(input_image, image_resolution))
        H, W, C = img.shape

        detected_map = apply_canny(img, 100, 200)
        detected_map = HWC3(detected_map)
        # detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
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

prompt = "Cardiomegaly with mild pulmonary vascular congestion."
input_img = cv2.imread("./test/test_01.jpg")
gen_img = process(input_img, prompt)
print(len(gen_img))
with open("./test/test_gen", "wb") as fp:   #Pickling
    pickle.dump(gen_img, fp)
#cv2.imwrite("./test/gen_out.jpg", gen_img[1])
for i in range(1,len(gen_img)):
    img = Image.fromarray(gen_img[i])
    image = AutoImage(img)
    print(image)
    #cv2.imshow(f'sample_{i}', 255-gen_img[i])
    #cv2.waitKey(0)
#cv2.destroyAllWindows()
print("Done.")
