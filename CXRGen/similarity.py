import cv2
import torchvision.transforms as T
from sentence_transformers import util
from groundingdino.util.inference import load_image
from PIL import Image

from torch.nn import CosineSimilarity


cos = CosineSimilarity(dim=0)
def imageEncoder(img):
    image_source, image = load_image(img)
    return image
def generateScore(image1, image2):
    test_img = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(image1)
    img2 = imageEncoder(image2)
    score = cos(img1, img2)
    return score

image1 = "test/4decce85-c6ede74e-7a8bc81c-e81edee9-5ec17116.jpg"
image2 = "test/samples/pt2025-06-13_14-00-13/gen_out_inv_sample4.jpg"

print(f"similarity Score: ", generateScore(image1, image2).mean())