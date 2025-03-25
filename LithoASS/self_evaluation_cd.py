import cv2
import numpy as np
import os
from scipy.stats import linregress


def extract_straight_lines_from_layout(layout):
    """
    从布局图像中提取直线部分。
    """
    binary = cv2.threshold(layout, 10, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_width, min_area = 50, 2500
    return [contour for contour in contours if
            max(cv2.boundingRect(contour)[2:]) >= min_width and cv2.contourArea(contour) >= min_area]


def calculate_iou(mask1, mask2):
    """
    计算两个区域的 IOU。
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0


def extract_cd_values(layout, iou_threshold=0.75, direction=1):
    """
    提取 CD 值并计算相关统计量。
    """
    straight_lines = extract_straight_lines_from_layout(layout)
    show_image = cv2.cvtColor(layout, cv2.COLOR_GRAY2BGR)
    cd_values_list = []
    cd_max_min_diffs = []  # 存储每个轮廓的 CD 值最大值减去最小值
    cd_fit_diffs = []  # 存储每个轮廓的 CD 值真实点减去拟合点的差值
    cd_variances = []  # 存储每个轮廓的 CD 值方差
    cd_average_abs_fit_diffs = []  # 存储每个轮廓的真实值减去拟合点的差值的绝对值的平均值

    for contour in straight_lines:
        x, y, w, h = cv2.boundingRect(contour)
        short_axis = 'vertical' if w > h else 'horizontal'
        mask = np.zeros_like(layout)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        mask1 = np.zeros_like(layout)
        cv2.rectangle(mask1, (x, y), (x + w, y + h), 255, -1)

        if short_axis == 'vertical':
            start_x = x + w // 2
            x1, x2 = start_x + direction, start_x - direction
            while x1 <= x + w and calculate_iou(mask[y:y + h, x1:x1 + 1], mask1[y:y + h, x1:x1 + 1]) >= iou_threshold:
                x1 += direction
            while x2 >= x and calculate_iou(mask[y:y + h, x2:x2 + 1], mask1[y:y + h, x2:x2 + 1]) >= iou_threshold:
                x2 -= direction
            x1, x2 = min(x1, x + w), max(x2, x)
            cv2.rectangle(show_image, (x2, y), (x1, y + h), (0, 0, 255), 2)
            contour_cd_values = [np.sum(mask[y:y + h, col:col + 1] > 0) for col in range(x2, x1)]
        else:
            start_y = y + h // 2
            y1, y2 = start_y + direction, start_y - direction
            while y1 <= y + h and calculate_iou(mask[y1:y1 + 1, x:x + w], mask1[y1:y1 + 1, x:x + w]) >= iou_threshold:
                y1 += direction
            while y2 >= y and calculate_iou(mask[y2:y2 + 1, x:x + w], mask1[y2:y2 + 1, x:x + w]) >= iou_threshold:
                y2 -= direction
            y1, y2 = min(y1, y + h), max(y2, y)
            cv2.rectangle(show_image, (x, y2), (x + w, y1), (0, 0, 255), 2)
            contour_cd_values = [np.sum(mask[row:row + 1, x:x + w] > 0) for row in range(y2, y1)]

        # 计算 CD 值的最大值减去最小值
        cd_max_min_diff = max(contour_cd_values) - min(contour_cd_values)
        cd_max_min_diffs.append(cd_max_min_diff)

        # 线性拟合 CD 值
        if len(contour_cd_values) > 1:
            indices = np.arange(len(contour_cd_values))
            slope, intercept, _, _, _ = linregress(indices, contour_cd_values)
            fit_line = slope * indices + intercept
            fit_diffs = contour_cd_values - fit_line
            # 计算方差时除以 n - 2
            variance = np.sum(fit_diffs ** 2) / (len(fit_diffs) - 2) if len(fit_diffs) > 2 else 0
            # 计算真实值减去拟合点的差值的绝对值的平均值
            average_abs_fit_diff = np.mean(np.abs(fit_diffs))
        else:
            fit_diffs = np.array([0])
            variance = 0
            average_abs_fit_diff = 0

        cd_fit_diffs.append(fit_diffs)
        cd_variances.append(variance)
        cd_average_abs_fit_diffs.append(average_abs_fit_diff)
        cd_values_list.append(contour_cd_values)

    return show_image, cd_values_list, cd_max_min_diffs, cd_fit_diffs, cd_variances, cd_average_abs_fit_diffs


def process_images(image_folder, output_folder):
    """
    处理一系列图像。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.bmp'))]
    all_cd_values = []
    all_cd_max_min_diffs = []
    all_cd_fit_diffs = []
    all_cd_variances = []
    all_cd_average_abs_fit_diffs = []

    for image_file in image_files:
        layout = cv2.imread(os.path.join(image_folder, image_file), cv2.IMREAD_GRAYSCALE)
        show_image, cd_values, cd_max_min_diffs, cd_fit_diffs, cd_variances, cd_average_abs_fit_diffs = extract_cd_values(
            layout)
        cv2.imwrite(os.path.join(output_folder, image_file), show_image)

        # 保存 CD 值、最大值减去最小值、拟合差值、方差和真实值减去拟合点的差值的绝对值的平均值
        all_cd_values.append(cd_values)
        all_cd_max_min_diffs.append(cd_max_min_diffs)
        all_cd_fit_diffs.append(cd_fit_diffs)
        all_cd_variances.append(cd_variances)
        all_cd_average_abs_fit_diffs.append(cd_average_abs_fit_diffs)

        # 保存每个轮廓的 CD 值及相关统计到 .txt 文件
        txt_output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_cd_values.txt")
        with open(txt_output_path, 'w') as f:
            for i, values in enumerate(cd_values):
                f.write(f"Contour {i + 1} CD values:\n{', '.join(map(str, values))}\n")
                f.write(f"CD Max-Min Diff: {cd_max_min_diffs[i]}\n")
                f.write(f"CD Fit Diffs: {', '.join(map(str, cd_fit_diffs[i]))}\n")
                f.write(f"CD Variance: {cd_variances[i]}\n")
                f.write(f"CD Average Abs Fit Diff: {cd_average_abs_fit_diffs[i]}\n\n")

        print(f"Processed {image_file}: proceed")

    # 保存所有图像的 CD 值统计量到一个 txt 文件
    stats_txt_path = os.path.join(output_folder, 'all_cd_stats.txt')
    with open(stats_txt_path, 'w') as f:
        for image_file, cd_values, cd_max_min_diffs, cd_fit_diffs, cd_variances, cd_average_abs_fit_diffs in zip(
                image_files, all_cd_values, all_cd_max_min_diffs, all_cd_fit_diffs, all_cd_variances,
                all_cd_average_abs_fit_diffs):
            f.write(f"Image: {image_file}\n")
            for i, (values, max_min_diff, fit_diffs, variance, average_abs_fit_diff) in enumerate(
                    zip(cd_values, cd_max_min_diffs, cd_fit_diffs, cd_variances, cd_average_abs_fit_diffs)):
                f.write(f"Contour {i + 1}:\n")
                f.write(f"CD Values: {', '.join(map(str, values))}\n")
                f.write(f"CD Max-Min Diff: {max_min_diff}\n")
                f.write(f"CD Fit Diffs: {', '.join(map(str, fit_diffs))}\n")
                f.write(f"CD Variance: {variance}\n")
                f.write(f"CD Average Abs Fit Diff: {average_abs_fit_diff}\n\n")

    return all_cd_values


if __name__ == '__main__':
    image_folder = 'seg_3_epoch3/seg_3_epoch3'
    output_folder = 'self_evaluation_cd'
    all_cd_values = process_images(image_folder, output_folder)
    print("All CD values processed.")
