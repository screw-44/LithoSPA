import copy

import torch
import cv2
import numpy as np

from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor
from LithoSeg.inference import load_paired_data
from LithoPP.inference import get_pp_contours, pp_contours2image
from LithoPP.model import EdgeNet as LithoPP
from LithoASS.self_evaluation_cd import assess_cd
from LithoASS.self_evaluation_edgepoint import assess_edgepoint


device = "cuda" if torch.cuda.is_available() else "cpu"

# LithoSEG
sam_checkpoint = "/Users/hexinyu/PycharmProjects/LithoSPA/experiment/exp3_epoch3.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# LithoPP
litho_pp = LithoPP().to(device)
litho_pp.load_state_dict(torch.load("./LithoPP/best_edge_model.pth", map_location=device))
litho_pp.eval()

seg_data = load_paired_data()

for data in tqdm(seg_data):
    # Use LithoSeg to Seg mask
    image = data["litho_image"]
    show_sem = copy.deepcopy(image)
    input_boxes = torch.tensor(data["boxes"]).to(device)
    predictor.set_image(image)

    transform_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transform_boxes, #None
        multimask_output=False,
    )

    gen_seg_mask = np.zeros((1024, 1024), dtype=np.uint8)
    for mask in masks:
        #print(mask.shape)
        gen_seg_mask[mask.cpu().numpy()[0] == 1] = 1

    # Use LithoPP to pinpoint edges
    contours, _ = cv2.findContours(gen_seg_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(show_sem, contours, -1, (0, 255, 0), 2)
    pp_contours = get_pp_contours(contours, litho_pp, image, show_sem)
    pp_mask = pp_contours2image(pp_contours, show_sem)

    # Use LithoASS to assessment lithography parameters
    pp_mask = cv2.cvtColor(pp_mask, cv2.COLOR_BGR2GRAY)
    results = assess_edgepoint(pp_mask)
    x_list, show_image_1, cd_values_1, first_point_diffs, last_point_diffs, first_point_fit_diffs, last_point_fit_diffs, first_point_fit_mean_diffs, last_point_fit_mean_diffs, contour_square_diffs_first_list, contour_square_diffs_last_list, contour_square_diff_means_list = results
    show_image_2, cd_values_2, cd_max_min_diffs, cd_fit_diffs, cd_variances, cd_average_abs_fit_diffs = assess_cd(pp_mask)

    cv2.imshow("sem", show_sem)
    cv2.imshow("pp_mask", pp_mask)
    cv2.imshow("show_image_1", show_image_1)
    cv2.imshow("show_image_2", show_image_2)
    cv2.waitKey(0)



