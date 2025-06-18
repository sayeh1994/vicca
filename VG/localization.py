'''
The model is trained on the architecture of the Grounding DINO with replacement of the image and text encoder.
The testing code is coming from GD model with minor changes in the configuration to be suited for CXRs.

# A code sample to run this script:
python test.py --weight_path="weights/checkpoint0399.pth" --image_path="38708899-5132e206-88cb58cf-d55a7065-6cbc983d.jpg"\
               --text_prompt="Cardiomegaly with mild pulmonary vascular congestion." --box_threshold=0.3 \
               --text_threshold=0.2 --plot_boxes
'''

from .groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import matplotlib.pyplot as plt
import supervision as sv
import torch
from torchvision.ops import box_convert
import numpy as np
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set Visual Grounding', add_help=False)
    parser.add_argument('--weight_path', type=str, default="weights/checkpoint_best_regular.pth", 
                        help="The path to the trained model")
    parser.add_argument('--image_path', type=str, 
                        help="The path to the image file.")
    parser.add_argument('--text_prompt', type=str, 
                        help="The text prompt.")
    parser.add_argument('--box_threshold', default=0.22, type=float)
    parser.add_argument('--text_threshold', default=0.2, type=float)
    parser.add_argument('--plot_boxes', action='store_true')
    return parser

def convert_boxes_to_numpy(boxes, image_source):
    h, w, _ = image_source.shape
    bbox = boxes * torch.Tensor([w, h, w, h])
    bbox = box_convert(boxes=bbox, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return bbox

def main(args):
    model = load_model("VG/groundingdino/config/GroundingDINO_SwinT_OGC.py", args.weight_path)

    IMAGE_PATH = args.image_path
    TEXT_PROMPT = args.text_prompt

    BOX_TRESHOLD = args.box_threshold
    TEXT_TRESHOLD = args.text_threshold

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    if args.plot_boxes:
        annotate_dict = dict(color=sv.ColorPalette.DEFAULT, thickness=2, text_thickness=1)

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases, bbox_annot=annotate_dict)
        plt.imshow(annotated_frame, cmap="gray")
        plt.axis('off')

    bbox = convert_boxes_to_numpy(boxes, image_source)
    # print(bbox, logits, phrases)
    return bbox, logits, phrases

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visual Grounding of CXR Image Prompt', parents=[get_args_parser()])
    args = parser.parse_args()
    bbox, logits, phrases = main(args)