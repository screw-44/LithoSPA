import torch
import torchvision
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from tqdm import tqdm

from utils import visualize_tensor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def load_paired_data(data_root="../sjw/layout2adi0907/train"):
    mask_path, ADI_path = data_root + "/layout", data_root + "/ADI"
    seg_data = []
    for mask_image_path in tqdm(glob.glob(mask_path+"/*")):
        file_name = os.path.basename(mask_image_path)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        # 反转图像
        mask_image = cv2.bitwise_not(mask_image)
        litho_image = cv2.imread(f"{ADI_path}/{file_name.split('.')[0]}.bmp")
        mask_image = cv2.resize(mask_image, (1024, 1024))
        litho_image = cv2.resize(litho_image, (1024, 1024))

        mask_image[0:15, :] = 0 # 把边缘set为0
        mask_image[-15:, :] = 0
        mask_image[:, 0:15] = 0
        mask_image[:, -15:] = 0

        contours, _ = cv2.findContours(mask_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(litho_image, (x, y), (x+w, y+h), (255, 0, 0), 4)
            #print(x, y, w, h)
            boxes.append([x, y, x+w, y+h]) # 左上和右下
        # cv2.imshow("mask", mask_image)
        # cv2.imshow("litho", litho_image)
        # cv2.waitKey(0)

        seg_infor = {
            "file_name" : file_name,
            "mask_image" : mask_image,
            "litho_image" : litho_image,
            "boxes": boxes
        }
        seg_data.append(seg_infor)
    return seg_data

if __name__ == "__main__":
    # 进入SAM操作部分

    from segment_anything import sam_model_registry, SamPredictor

    device = "cuda:0"
    # sam_checkpoint = "checkpoint/sam_vit_b_01ec64.pth"
    sam_checkpoint = "experiment/ss_pure_contrast_fbase_exp1/epoch1.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    seg_data = load_paired_data()

    for seg_infor in tqdm(seg_data):
        image = seg_infor["litho_image"]
        input_boxes = torch.tensor(seg_infor["boxes"]).to(device)
        predictor.set_image(image)

        transform_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transform_boxes, #None
            multimask_output=False,
        )

        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box in input_boxes:
        #     show_box(box.cpu().numpy(), plt.gca())
        # plt.axis('off')
        # plt.show()

        gen_seg_mask = np.zeros((1024, 1024), dtype=np.uint8)
        for mask in masks:
            #print(mask.shape)
            gen_seg_mask[mask.cpu().numpy()[0] == 1] = 255

        data_dir = "./dataset/ss_pure_contrast_fbase_exp1"
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        file_name = seg_infor["file_name"].split('.')[0]
        cv2.imwrite(f"{data_dir}/{file_name}.jpg", gen_seg_mask)




