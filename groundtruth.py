import glob
import os
import random
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2 as cv
import torch
from train import *
from torch.multiprocessing import Pool

from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression, RANSACRegressor


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def line_from_points(point1, point2):
    a = (point2[1]-point1[1])/(point2[0]-point1[0] + 0.0001)
    b = point1[1] - a * point1[0]
    return a, b


# get solid cross-whole-image lines from the layout image，很成功，都能把贯穿的线提取出来。
def get_solid_lines_from_layout_image(layout):
    binary = cv.threshold(layout, 10, 255, cv.THRESH_BINARY_INV)[1]
    # 使用形态学闭运算填充断点
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

    # 实际图像中有一些白边在边缘，全部改为黑色
    binary[0:10, :] = 0
    binary[-10:, :] = 0
    binary[:, 0:10] = 0
    binary[:, -10:] = 0

    _contours, _ = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    min_width, max_height = 1000, 500  # the image width is 1024
    cross_line_bbox = []
    cross_line_contour = []
    for contour in _contours:
        x, y, w, h = cv.boundingRect(contour)
        # 太靠近边缘，剔除
        if y < 20 or (y + h) > 1000:
            continue
        if w >= min_width and h < max_height:
            cross_line_contour.append(contour)
            cross_line_bbox.append([x, y, w, h])
    return cross_line_bbox



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EdgeNet().to(device)
model.load_state_dict(torch.load("best_edge_model.pth", map_location=device))
model.eval()  

def get_precise_contour(top_contour, bottom_contour, grad, sem, model, device, output_folder='brightness_profiles', image_name='image'):
    image_output_folder = os.path.join(output_folder, image_name)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    new_seg_path = os.path.join(image_output_folder, f"{image_name}_offset.png")
    
   
    if os.path.exists(new_seg_path):
        new_seg_image = cv2.imread(new_seg_path, cv2.IMREAD_GRAYSCALE)  
    else:
        new_seg_image = np.zeros_like(sem)  

    def process_single_contour(contour):
        process_contour = [point[0] for point in contour]  
        predicted_contour = np.empty((0, 2), dtype=np.int32)

        for index in range(len(process_contour)):
            current_point = process_contour[index]
            direction_vector = line_from_points(process_contour[index], process_contour[(index + 1) % len(process_contour)])

            # 计算法线方向
            normal_vector = (-1 / (direction_vector[0] + 1e-4), current_point[1] + 1 / (direction_vector[0] + 1e-4) * current_point[0])

            # 扫描亮度值
            scan_size = 15
            scan_grad = []
            for shift in range(-scan_size, scan_size):
                if abs(normal_vector[0]) >= 1:
                    y = current_point[1] + shift
                    x = int((y - normal_vector[1]) / normal_vector[0])
                else:
                    x = current_point[0] + shift
                    y = int(normal_vector[0] * x + normal_vector[1])
                if 0 < x < 1024 and 0 < y < 1024:
                    scan_grad.append([x, y, grad[y, x], sem[y, x]])

            if not scan_grad or len(scan_grad) < 30:
                continue

            scan_grad = np.array(scan_grad)
            brightness_column = scan_grad[:, 3]
            brightness_filtered = gaussian_filter1d(brightness_column, 5)

            # 预测偏移
            with torch.no_grad():
                brightness_tensor = torch.tensor(brightness_filtered, dtype=torch.float32).unsqueeze(0).to(device)
                predictions = model(brightness_tensor).cpu().numpy()[0]  
                center_offset = int(round(predictions[2]))  # 偏移量

            # 沿法线方向移动 center_offset 个像素
            if abs(normal_vector[0]) >= 1:
                new_y = current_point[1] + center_offset
                new_x = int((new_y - normal_vector[1]) / normal_vector[0])
            else:
                new_x = current_point[0] + center_offset
                new_y = int(normal_vector[0] * new_x + normal_vector[1])

    
            if 0 <= new_x < 1024 and 0 <= new_y < 1024:
                new_point = np.array([new_x, new_y], dtype=np.int32).reshape(1, 2)
                if len(predicted_contour) == 0:
                    predicted_contour = np.array([[new_x, new_y]], dtype=np.int32)
                else:
                    predicted_contour = np.vstack([predicted_contour, new_point])

        return predicted_contour

    predicted_top_contour = process_single_contour(top_contour)
    predicted_bottom_contour = process_single_contour(bottom_contour)
    full_predicted_contour = np.vstack([predicted_top_contour, predicted_bottom_contour])

    if len(full_predicted_contour) > 0:
        full_predicted_contour = full_predicted_contour.reshape(-1, 1, 2)  # 转换为OpenCV所需的形状
        cv2.polylines(new_seg_image, [full_predicted_contour], isClosed=False, color=255, thickness=2) 

    cv2.imwrite(new_seg_path, new_seg_image)
    print(f"Updated contour shape: {full_predicted_contour.shape} -> Saved at {new_seg_path}")
     
def fit_line(contour):
    # 使用 RANSAC 拟合直线
    random_points = random.sample(contour.tolist(), min(50, len(contour)))  # 增加采样点数量
    x = np.array([point[0][0] for point in random_points])
    y = np.array([point[0][1] for point in random_points])
    model = RANSACRegressor(LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    return model


def sem_process(layout, sem):
    # pre-calcualte the whole image gradient
    grad_x = cv.Sobel(sem, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv.Sobel(sem, cv2.CV_64F, 0, 1, ksize=5)
    grad_sem = np.sqrt(grad_x ** 2 + grad_y ** 2)

    sem = cv.threshold(sem, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    sem = cv.morphologyEx(sem, cv.MORPH_CLOSE, (40, 40))

    contours, hierarchies = cv.findContours(sem, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        # 点数太小直接过滤
        if len(contour) < 50:
            continue
        # 面积太小直接过滤
        contour_area = cv.contourArea(contour)
        if contour_area < 200: continue
        # 过滤contour的高度差，去除只有一条线，无法计算CD的情况
        y_list = [point[0][1] for point in contour]
        smallest_mean = np.mean(np.partition(y_list, 5)[:5])
        biggest_mean = np.mean(np.partition(y_list, -5)[-5:])
        if (biggest_mean - smallest_mean) < 15:  # 单根线也不满足
            continue
        filtered_contours.append(contour)

    filtered_contour_bbox = [[cv.boundingRect(contour), contour] for contour in filtered_contours]
    cross_line_bbox = get_solid_lines_from_layout_image(layout)
    paired_filtered_contours = []
    for bbox in cross_line_bbox:
        top_left = (bbox[0], bbox[1])
        bottom_left = (bbox[0], bbox[1] + bbox[3])
        _top_contour = sorted(filtered_contour_bbox,
                              key=lambda x: ((top_left[0] - x[0][0]) ** 2 + (top_left[1] - x[0][1]) ** 2) ** 0.5)[0][1]
        _bottom_contour = sorted(filtered_contour_bbox,
                                 key=lambda x: ((bottom_left[0] - x[0][0]) ** 2 + (
                                             bottom_left[1] - (x[0][1] + x[0][3])) ** 2) ** 0.5)[0][1]
        paired_filtered_contours.append([_top_contour, _bottom_contour])

    return grad_sem, paired_filtered_contours


def get_spec(sem_grad, sem, paired_filtered_contours, show_sem, image_name):
    loss_list = []
    
    output_folder = "segmented_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_output_path = os.path.join(output_folder, f"{image_name}.png")
    
    result_image = show_sem.copy()  
    for _top_contour, _bottom_contour in paired_filtered_contours:
        get_precise_contour(_top_contour, _bottom_contour, sem_grad, sem, model, device, output_folder, image_name)
    
    cv2.imwrite(image_output_path, result_image)
    print(f"Final processed image saved: {image_output_path}")
    
    return loss_list

def wulala(layout, sem):
    layout = cv.imread(layout)
    sem = cv.imread(sem)
    sem = cv.cvtColor(sem, cv.COLOR_BGR2GRAY)
    show_sem = sem
    layout = cv.resize(layout, (1024,1024))
    sem_grad, paired_filtered_contours = sem_process(layout, sem.copy())
    loss_list = get_spec(sem_grad, sem, paired_filtered_contours, show_sem)


if __name__ == '__main__':
    # sem_paths = "/Users/hexinyu/PycharmProjects/sjw/layout2adi0907/train/ADI/*"
    sem_paths = "/home/robot/personal/segment/project/sem/*"
    sem_paths = glob.glob(sem_paths)
    sem_paths_dict = {os.path.basename(path[:-4]): path for path in sem_paths}

    # layout_paths = "/Users/hexinyu/PycharmProjects/sjw/layout2adi0907/train/layout/*"
    layout_paths = "/home/robot/personal/segment/project/segmentation/*"
    layout_paths = glob.glob(layout_paths)
    layout_paths_dict = {os.path.basename(path[:-4]): path for path in layout_paths}

    # add paired data for the detection
    l2s_data = []
    for layout_path in layout_paths_dict:
        if layout_path in sem_paths_dict:
            l2s_data.append([layout_paths_dict[layout_path], sem_paths_dict[layout_path]])

    for layout_path, sem_path in l2s_data:  # 修改为 layout_path 和 sem_path
        layout = cv.imread(layout_path)  # 使用 layout_path 读取图像
        sem = cv.imread(sem_path)  # 使用 sem_path 读取图像
        show_sem = sem
        sem = cv.cvtColor(sem, cv.COLOR_BGR2GRAY)
        layout = cv.resize(layout, (1024, 1024))
        layout = cv.cvtColor(layout, cv.COLOR_BGR2GRAY)
        # 获取当前时间的秒数
        t_seconds = time.time()
        # 将秒数转换为纳秒
        t_ns = int(t_seconds * 1e9)
        sem_grad, paired_filtered_contours = sem_process(layout, sem.copy())
        print("time", (time.time() * 1e9 - t_ns) // 1e6)
        t_seconds = time.time()
        # 将秒数转换为纳秒
        t_ns = int(t_seconds * 1e9)
        image_name = os.path.basename(sem_path)[:-4]  # 使用 sem_path 获取图片名称，去掉扩展名
        loss_list = get_spec(sem_grad, sem, paired_filtered_contours, show_sem, image_name)
        print("time", (time.time() * 1e9 - t_ns) // 1e6)

        # 11.17 测试结果显示multi-process无法加速计算，进程管理的成本大于计算成本。同时通过查阅资料可知，多线程在不等待的计算场景下，没有加速效果。
        # 下一步优化应该在矩阵计算和计算量减少的方向上。目前先写简易loss
        # t_ns = time.time_ns()
