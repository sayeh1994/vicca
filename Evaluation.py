"""
Input: image and text
Middle output: bbox (VG), Gen Image and similarity score (CXRGen), Shift_x&y (DETR)
Output: Localization Score, Reliability Score

python inference.py \
    --image_path VG/38708899-5132e206-88cb58cf-d55a7065-6cbc983d.jpg \
    --text_prompt "Cardiomegaly with mild pulmonary vascular congestion."

"""
import pandas as pd
import numpy as np
import time
import cv2
import sys
import argparse
from ast import literal_eval

sys.path.append('/home/gholipos-admin/Desktop/Thesis/Training_Code/VICCA/')
sys.path.append('/home/gholipos-admin/Desktop/Thesis/Training_Code/jointReportImage/R2Gen/')

from CXRGen import sample_generation
from DETR import svc
from DETR.arguments import get_args_parser as get_detr_args_parser
from VG import localization
from ssim import ssim


from Entity_Extract.EntityExtractorv2 import medical_term
from Entity_Extract.SimMetric import sim_metric
from CheXbert.src.label import label
from nltk import tokenize
from tqdm import tqdm


path_list = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
            'Support Devices', 'No Finding']


def chexbert_entity_extraction(text):
    sentences = list(set(tokenize.sent_tokenize(text)))
    new_impression_entity = []
    for sentence in sentences:
        sentence = sentence.replace('\n',' ')
        sentence = sentence.replace('\s+',' ')
        pathology = np.array(label("CheXbert/checkpoint/chexbert.pth", sentence)).T[0]
        if pathology[-1]==1 or len(list(set(pathology)))==1 or not any(e==1 for e in pathology):
            pass
        else:
            entity_list = medical_term(sentence)
            new_impression_entity.append(" ".join(entity_list))
    return new_impression_entity   

def chexbert_pathology(text):
    sentences = list(set(tokenize.sent_tokenize(text)))
    path_dict = []
    for sentence in sentences:
        sentence = sentence.replace('\n',' ')
        sentence = sentence.replace('\s+',' ')
        pathology = np.array(label("CheXbert/checkpoint/chexbert.pth", sentence)).T[0]
        if pathology[-1]==1 or len(list(set(pathology)))==1 or not any(e==1 for e in pathology):
            pass
        else:
            indice = [i for i, e in enumerate(pathology) if e==1]
            for ind in indice:
                path_dict.append(path_list[ind])
    return path_dict

def MCSE_score(Reference_Text, Candidate_Text):
    Reference_entities = medical_term(Reference_Text)
    Candidate_entities = medical_term(Candidate_Text)

    semantic_score = sim_metric(Reference_entities, Candidate_entities)
    return semantic_score


def get_args_parser():
    parser = argparse.ArgumentParser('Set the Input', add_help=True)
    parser.add_argument('--weight_path_gencxr', type=str, default="CXRGen/checkpoints/cn_d25ofd18_epoch-v18.pth", 
                        help="Path to the CXR generation trained model")
    parser.add_argument('--weight_path_vg', type=str, default="VG/weights/checkpoint0399.pth", 
                        help="Path to the Visual Grounding trained model")
    # parser.add_argument('--image_path', type=str, required=True,
                        # help="Path to the input image file.")
    # parser.add_argument('--text_prompt', type=str, required=True,
    #                     help="Text prompt describing pathology.")
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
    args = parser.parse_args([])
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

    meta_data = pd.read_csv("/home/gholipos-admin/Desktop/Thesis/Training_Code/jointReportImage/R2Gen/MIMIC_CXR/mimic-cxr-2.0.0-metadata.csv")
    pred = pd.read_csv("/home/gholipos-admin/Desktop/Thesis/Training_Code/jointReportImage/R2Gen/results/pred-org-model.csv", sep="\t",
                       header=None, names=["id", "report"])
    ref = pd.read_csv("/home/gholipos-admin/Desktop/Thesis/Training_Code/jointReportImage/R2Gen/results/ref-org-model.csv", sep="\t",
                       header=None, names=["id", "report"])
    
    general_path = "/home/gholipos-admin/Desktop/Thesis/Training_Code/jointReportImage/R2Gen/data/mimic_cxr_test/files/"
    store_dict={"dicom_id":[], 
                "chexbert_pred":[], 
                "chexbert_ref":[], 
                "MCSE":[], 
                "Localization_boxes": [],
                "Localization_logits": [],
                "Reliability_score": [],
                "predicted_report": [],
                "original_report": [],
                "file_path":[]}

    for ind, im_id in enumerate(pred["id"]):
        indices = meta_data[meta_data['dicom_id'].astype(str).str.contains(im_id, case=False, na=False)].index
        if meta_data["ViewPosition"][indices[0]] in ["PA", "AP"]:
            im_path=general_path+f"p{str(meta_data['subject_id'][indices[0]])[:2]}/"+f"p{meta_data['subject_id'][indices[0]]}/"+f"s{meta_data['study_id'][indices[0]]}/"+im_id+".jpg"

            pathology_pred = chexbert_pathology(pred["report"][ind])
            pathology_ref = chexbert_pathology(ref["report"][ind])
            if pathology_pred:
                mcse = MCSE_score(ref["report"][ind], pred["report"][ind])
                args = get_args_parser().parse_args()

                gen_cxr(args.weight_path_gencxr, im_path, pred["report"][ind], args.num_samples, args.output_path)
                time.sleep(4)  # ensure outputs are written

                df = pd.read_csv(args.output_path + "info_path_similarity.csv")
                sim_ratios = [extract_tensor(val) for val in df["similarity_rate"]]
                max_sim_index = sim_ratios.index(max(sim_ratios))
                max_sim_gen_path = df["gen_sample_path"][max_sim_index]

                sx, sy = cal_shift(im_path, max_sim_gen_path)

                boxes, logits, phrases = get_local_bbox(
                    args.weight_path_vg,
                    im_path,
                    pred["report"][ind],
                    args.box_threshold,
                    args.text_threshold
                )
                print("Boxes:", boxes)
                print("Phrases:", phrases)

                image_org_cv = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
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

                # if ssim_scores:
                #     print("SSIM scores per box:", ssim_scores)
                #     print("Localization Detection Scores per bbox:", boxes, logits)
                #     # print("Average SSIM (Localization Score):", sum(ssim_scores) / len(ssim_scores))
                # else:
                #     print("No valid SSIM scores (e.g., mismatched shapes or empty ROIs).")

                store_dict["dicom_id"].append(im_id)
                store_dict["chexbert_pred"].append(pathology_pred)
                store_dict["chexbert_ref"].append(pathology_ref)
                store_dict["MCSE"].append(mcse)
                store_dict["Localization_boxes"].append(boxes)
                store_dict["Localization_logits"].append(logits)
                store_dict["Reliability_score"].append(ssim_scores)
                store_dict["predicted_report"].append(pred["report"][ind])
                store_dict["original_report"].append(ref["report"][ind])
                store_dict["file_path"].append(im_path)
            else:
                store_dict["dicom_id"].append(im_id)
                store_dict["chexbert_pred"].append(pathology_pred)
                store_dict["chexbert_ref"].append(pathology_ref)
                store_dict["MCSE"].append(mcse)
                store_dict["Localization_boxes"].append([])
                store_dict["Localization_logits"].append([])
                store_dict["Reliability_score"].append([])
                store_dict["predicted_report"].append(pred["report"][ind])
                store_dict["original_report"].append(ref["report"][ind])
                store_dict["file_path"].append(im_path)

    df = pd.DataFrame(store_dict)
    df.to_csv('vicca_r2gen.csv', index=False)
    # # Save the DataFrame to a CSV file
    # try:
        
    # except:
    #     print("The Pandas has problem of the length.")
    #     with open('dice_value_gentest_2.json', 'w') as json_file:
    #         json.dump(dice_dict, json_file)
    #     print("The JSON file saved instead.")

