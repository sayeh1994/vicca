'''
python DETR/test.py --img_org="CXR-Gen/test/4decce85-c6ede74e-7a8bc81c-e81edee9-5ec17116.jpg"\\ 
                    --img_gen="CXR-Gen/test/samples/pt2025-06-13_14-45-52/gen_out_inv_sample1.jpg"
'''
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision
from torch import nn

from torchvision.models import resnet50
import torchvision.transforms as T
from models import build_model

import argparse
from arguments import get_args_parser

torch.set_grad_enabled(False);
CLASSES = ['right lung', 'right upper lung zone', 'right mid lung zone', 'right lower lung zone', 'right hilar structures', 
           'right apical zone','right costophrenic angle', 'right cardiophrenic angle','right hemidiaphragm',
           'left lung','left upper lung zone','left mid lung zone','left lower lung zone','left hilar structures',
           'left apical zone','left costophrenic angle','left hemidiaphragm','trachea','spine','right clavicle',
           'left clavicle','aortic arch','mediastinum','upper mediastinum','svc','cardiac silhouette',
           'left cardiac silhouette','right cardiac silhouette','cavoatrial junction','right atrium','descending aorta',
           'carina','left upper abdomen','right upper abdomen','abdomen','left cardiophrenic angle']

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled


def read_image(image_path):
    return Image.open(image_path).convert('RGB'), cv2.imread(image_path)

def main(args):
    
    model, criterion, postprocessors = build_model(args)
    state_dict = torch.load(args.read_checkpoint)
    model.load_state_dict(state_dict["model"])
    model.eval()
    
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_org_pl, image_org_cv = read_image(args.img_org)
    image_gen_pl, image_gen_cv = read_image(args.img_gen)
    
    scores_org, boxes_org = detect(image_org_pl, model, transform)
    scores_gen, boxes_gen = detect(image_gen_pl, model, transform)
    
    class_dict = {cl:0 for cl in CLASSES}
    for p, (x1, y1, w, h) in zip(scores_org, boxes_org.tolist()):
        cl = p.argmax()
        text = CLASSES[cl]               
        if CLASSES[cl] == 'svc':
            svc_org_bbox = [int(x1), int(y1), int(w), int(h)]
    
    for p, (x1, y1, w, h) in zip(scores_gen, boxes_gen.tolist()):
        cl = p.argmax()               
        if CLASSES[cl] == 'svc':
            svc_gen_bbox = [int(x1), int(y1), int(w), int(h)]
    
    shift_x , shift_y = svc_gen_bbox[0] - svc_org_bbox[0], svc_gen_bbox[1] - svc_org_bbox[1]
    print(shift_x , shift_y)
            
    # for bbox in xyxy:
    #     bbox1 = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
    #     bbox2 = [
    #                 bbox[0] + shift_x,  # x-coordinate
    #                 bbox[1] + shift_y,  # y-coordinate
    #                 bbox[2] - bbox[0],            # width remains the same
    #                 bbox[3] - bbox[1]          # height remains the same
    #                                 ]
    return shift_x , shift_y
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
