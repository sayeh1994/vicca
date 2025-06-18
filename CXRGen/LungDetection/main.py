import torch
import torchvision
# import cv2
from tqdm import tqdm

import pandas as pd
import numpy as np

from pathlib import Path
from PIL import Image

from CXRGen.LungDetection.src.models import PretrainedUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
models_folder = Path("CXRGen/LungDetection/models")

unet = PretrainedUNet(
    in_channels=1,
    out_channels=2, 
    batch_norm=True, 
    upscale_mode="bilinear"
)

model_name = "unet-6v.pt"
unet.load_state_dict(torch.load(models_folder / model_name, map_location=torch.device("cpu")))
unet.to(device)
unet.eval();

def segment(path, model, device):
    origin = Image.open(path).convert("P")
    origin = torchvision.transforms.functional.resize(origin, (512, 512))
    origin = torchvision.transforms.functional.to_tensor(origin) - 0.5
    
    with torch.no_grad():
        origin = torch.stack([origin])
        origin = origin.to(device)
        out = model(origin)
        softmax = torch.nn.functional.log_softmax(out, dim=1)
        out = torch.argmax(softmax, dim=1)
        
        origin = origin[0].to("cpu")
        out = out[0].to("cpu")
        return out

def lungsegment(img_path):
    out_image = segment(img_path, unet, device).cpu().numpy()
    out_image = Image.fromarray((out_image * 255).astype(np.uint8))
    out_image.save(img_path.split(".jpg")[0]+'-mask.jpg')
    return np.array(out_image).copy()

if __name__ == '__main__':
    lungsegment(img_path)
