"""
Input: image and text
Middle output: bbox (VG), Gen Image and similarity score (CXRGen), Shift_x&y (DETR)
Output: Localization Score, Reliability Score

python inference.py \
    --image_path VG/38708899-5132e206-88cb58cf-d55a7065-6cbc983d.jpg \
    --text_prompt "Cardiomegaly with mild pulmonary vascular congestion."

"""
import pandas as pd
import time
import cv2
import sys
import argparse
from ast import literal_eval

sys.path.append('/path/to/VICCA')

from CXRGen import sample_generation
from DETR import svc
from DETR.arguments import get_args_parser as get_detr_args_parser
from VG import localization
from ssim import ssim


def get_args_parser():
    parser = argparse.ArgumentParser('Set the Input', add_help=True)
    parser.add_argument('--weight_path_gencxr', type=str, default="CXRGen/checkpoints/cn_d25ofd18_epoch-v18.pth", 
                        help="Path to the CXR generation trained model")
    parser.add_argument('--weight_path_vg', type=str, default="VG/weights/checkpoint0399.pth", 
                        help="Path to the Visual Grounding trained model")
    parser.add_argument('--image_path', type=str, required=True,
                        help="Path to the input image file.")
    parser.add_argument('--text_prompt', type=str, required=True,
                        help="Text prompt describing pathology.")
    parser.add_argument('--box_threshold', default=0.2, type=float, help="Box threshold for VG")
    parser.add_argument('--text_threshold', default=0.2, type=float, help="Text threshold for VG")
    parser.add_argument('--num_samples', type=int, default=4, help="Number of generated image samples.")
    parser.add_argument('--output_path', type=str, default="CXRGen/test/samples/output/",
                        help="Path to save generated files.")
    return parser


def extract_tensor(value):
    cleaned_value = value.replace('tensor(', '').replace(')', '')
    return literal_eval(cleaned_value)


def gen_cxr(weight_path, image_path, text_prompt, num_samples, output_path):
    parser = sample_generation.get_args_parser()
    args = parser.parse_args([])  # Use empty args to override CLI
    args.weight_path = weight_path
    args.image_path = image_path
    args.text_prompt = text_prompt
    args.num_samples = num_samples
    args.output_path = output_path
    sample_generation.main(args)


def cal_shift(img_org_path, img_gen_path):
    parser = get_detr_args_parser()
    args = parser.parse_args([])
    args.img_org = img_org_path
    args.img_gen = img_gen_path
    shift_x, shift_y = svc.main(args)
    return shift_x, shift_y


def get_local_bbox(weight_path, image_path, text_prompt, box_threshold, text_threshold):
    parser = localization.get_args_parser()
    args = parser.parse_args([])
    args.weight_path = weight_path
    args.image_path = image_path
    args.text_prompt = text_prompt
    args.box_threshold = box_threshold
    args.text_threshold = text_threshold
    bbox, logits, phrases = localization.main(args)
    return bbox, logits, phrases


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    gen_cxr(args.weight_path_gencxr, args.image_path, args.text_prompt, args.num_samples, args.output_path)
    time.sleep(10)  # ensure outputs are written

    df = pd.read_csv(args.output_path + "info_path_similarity.csv")
    sim_ratios = [extract_tensor(val) for val in df["similarity_rate"]]
    max_sim_index = sim_ratios.index(max(sim_ratios))
    max_sim_gen_path = df["gen_sample_path"][max_sim_index]

    sx, sy = cal_shift(args.image_path, max_sim_gen_path)

    boxes, logits, phrases = get_local_bbox(
        args.weight_path_vg,
        args.image_path,
        args.text_prompt,
        args.box_threshold,
        args.text_threshold
    )
    print("Boxes:", boxes)
    print("Phrases:", phrases)

    image_org_cv = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    image_gen_cv = cv2.imread(max_sim_gen_path, cv2.IMREAD_GRAYSCALE)

    ssim_scores = []
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        bbox1 = [x1, y1, x2 - x1, y2 - y1]
        bbox2 = [x1 + sx, y1 + sy, x2 - x1, y2 - y1]

        bx1, by1, bw1, bh1 = [int(val) for val in bbox1]
        bx2, by2, bw2, bh2 = [int(val) for val in bbox2]

        roi_org = image_org_cv[by1:by1 + bh1, bx1:bx1 + bw1]
        roi_gen = image_gen_cv[by2:by2 + bh2, bx2:bx2 + bw2]

        if roi_org.shape == roi_gen.shape and roi_org.size > 0:
            score = ssim(roi_org, roi_gen)
            ssim_scores.append(score)

    if ssim_scores:
        print("SSIM scores per box:", ssim_scores)
        print("Localization Detection Scores per bbox:", boxes, logits)
        # print("Average SSIM (Localization Score):", sum(ssim_scores) / len(ssim_scores))
    else:
        print("No valid SSIM scores (e.g., mismatched shapes or empty ROIs).")
