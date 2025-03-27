import copy
import os
import cv2
import random
import numpy as np
import torch.backends.mps

from scipy.ndimage import gaussian_filter1d
from glob import glob
from tqdm import tqdm
from LithoPP.model import EdgeNet


def get_normal_vector(contour, index):
    if len(contour) < 2:
        return None  # 轮廓点太少，不计算法线

    # 获取局部窗口内的点，提高法向量稳定性
    window_size = 5
    start = max(0, index - window_size)
    end = min(len(contour), index + window_size)
    local_points = np.array(contour[start:end], dtype=np.float32)

    # 使用 cv2.fitLine 拟合局部直线
    [vx, vy, x0, y0] = cv2.fitLine(local_points, cv2.DIST_L2, 0, 0.01, 0.01)

    # 计算法向量（旋转90°）
    normal_vector = np.array([-vy, vx]).flatten()
    return normal_vector

def get_pp_contours(contours, model, sem, show_sem, device="cpu"):
    pp_contours = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 50 or contour.shape[0] < 50:
            # print("contour size too small")
            continue
        cv2.drawContours(show_sem, contours, i, (0, 255, 0), 3)
        # process
        pp_contour = []
        for i, point in enumerate(contour):
            point = point[0]
            # 如果point过于接近边缘，直接给定边缘的数值
            x, y = point
            if x <= 1 or x >= 1022 or y <=1 or y >= 1022:
                pp_contour.append([x, y])
                continue

            # find normal vector
            normal_vector = get_normal_vector(contour, i)
            # 改成两阶段，先找到区域中最大的亮度，然后再从这里开始scan前后30个
            # 阶段1: 找到区域中最大亮度
            scan_size, scan_grad = 15, []
            for shift in range(-scan_size, scan_size):
                offset = normal_vector * shift
                x, y = int(point[0] + offset[0]), int(point[1] + offset[1])  # 确保是整数
                if 0 <= x < 1024 and 0 <= y < 1024:  # 避免索引超出范围
                    scan_grad.append([x, y, sem[y, x][0]])
                else:
                    scan_grad.append([x, y, -1])  # TODO：超出索引范围就给-1，就设定直接是边缘情况
            scan_grad = np.array(scan_grad)
            brightness = scan_grad[:, 2]
            brightness_filtered = gaussian_filter1d(brightness, 0.1)
            max_brightness_point = scan_grad[brightness_filtered.argmax()]
            # 阶段2: 在最亮的点的基础上scan
            scan_size, scan_grad = 15, []
            for shift in range(-scan_size, scan_size):
                offset = normal_vector * shift
                x, y = int(max_brightness_point[0] + offset[0]), int(max_brightness_point[1] + offset[1])  # 确保是整数
                if 0 <= x < 1024 and 0 <= y < 1024:  # 避免索引超出范围
                    scan_grad.append([x, y, sem[y, x][0]])
                else:
                    scan_grad.append([x, y, -1])  # 超出索引范围就给-1来填充
            scan_grad = np.array(scan_grad)
            brightness = scan_grad[:, 2]
            brightness_filtered = gaussian_filter1d(brightness, 2)
            # predict offset
            with torch.no_grad():
                brightness_tensor = torch.tensor(brightness_filtered, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
                predictions = model(brightness_tensor).cpu().numpy()[0] * scan_size * 2
                center_offset = int(round(predictions[1])) - scan_size
            # 沿法线方向移动 center_offset 个像素
            offset = normal_vector * center_offset
            new_x, new_y = int(max_brightness_point[0] + offset[0]), int(max_brightness_point[1] + offset[1])
            if 0 <= new_x < 1024 and 0 <= new_y < 1024:
                pp_contour.append([new_x, new_y])

            cv2.circle(show_sem, [scan_grad[0][0], scan_grad[0][1]], radius=2, color=(0, 0, 255), thickness=1)
            cv2.circle(show_sem, [scan_grad[-1][0], scan_grad[-1][1]], radius=2, color=(0, 0, 255), thickness=1)
            # cv2.imshow("sem", sem)
            # cv2.waitKey(0)
        if pp_contour:
            pp_contours.append(pp_contour)

    return pp_contours


def pp_contours2image(pp_contours, show_sem):
    pp_mask = np.zeros_like(show_sem)
    contours_center = []
    for contour in pp_contours:
        for i in range(len(contour) - 1):
            cv2.line(show_sem, contour[i], contour[i + 1], (255, 0, 0), 1)
            cv2.line(pp_mask, contour[i], contour[i + 1], (255, 255, 255), thickness=3)
        cv2.line(show_sem, contour[-1], contour[0], (255, 0, 0), 1)
        cv2.line(pp_mask, contour[-1], contour[0], (255, 255, 255), thickness=3)
        x_contour, y_contour = [x for x, y in contour], [y for x, y in contour]
        # 强假定：一定会在x的中间有y
        mid_x = (min(x_contour) + max(x_contour)) // 2
        mid_ys = [y for x, y in contour if abs( x - mid_x) < 5]
        contours_center.append([mid_x, (min(mid_ys) + max(mid_ys)) // 2])

    h, w = pp_mask.shape[:2]
    mask = np.zeros((h + 2, w + 2, 1), np.uint8)
    for center in contours_center:
        cv2.floodFill(pp_mask, mask, center, (255, 255, 255), (0, 0, 0), (0, 0, 0))
        cv2.circle(show_sem, center, 7, [255, 0, 255], -1)
    return pp_mask

if __name__ == "__main__":
    sem_paths = "/Users/hexinyu/PycharmProjects/sjw/layout2adi0907/train/ADI/*"
    seg_mask_base_path = "../dataset/seg_4_epoch3/"
    paired_mask_sem_paths = [[seg_mask_base_path + os.path.basename(sem_path)[:-4] + ".jpg", sem_path] for sem_path in glob(sem_paths)]

    device = torch.device("cpu") # "mps:0" if torch.backends.mps.is_available() else "cpu")
    model = EdgeNet().to(device)
    model.load_state_dict(torch.load("best_edge_model.pth", map_location=device))
    model.eval()

    for mask_path, sem_path in tqdm(paired_mask_sem_paths):
        mask, sem = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), cv2.imread(sem_path)
        show_sem = copy.copy(sem)
        # extract all contours
        _, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        pp_contours = get_pp_contours(contours, model, sem, show_sem)

        pp_mask = pp_contours2image(pp_contours, show_sem)

        file_name = mask_path.split("/")[-1]
        cv2.imwrite(f'../experiment/seg_gt_3_27/pp_mask/{file_name}', pp_mask)
        cv2.imwrite(f'../experiment/seg_gt_3_27/draw_sem/{file_name}', show_sem)
        #cv2.imshow("pp_mask", pp_mask)
        #cv2.imshow("sem", show_sem)
        #cv2.waitKey(0)

        # TODO：batch-size调大的模型推理，应该能极大的提升分割速度。


