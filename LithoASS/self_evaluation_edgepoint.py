import cv2
import numpy as np
import os
from scipy.stats import linregress


def extract_straight_lines(layout):
    """
    从布局图像中提取直线部分。
    """
    # 二值化处理，保留白色部分
    binary = cv2.threshold(layout, 10, 255, cv2.THRESH_BINARY)[1]
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选长直线和面积足够大的轮廓
    min_width = 50
    min_area = 2500
    straight_lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        width = max(w, h)
        area = w * h
        if width >= min_width and area >= min_area:
            straight_lines.append(contour)
    return straight_lines


def calculate_iou(mask1, mask2):
    """
    计算两个区域的 IOU。
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0


def measure_cd_values(layout, iou_threshold=0.75, direction=1):
    x_list = []
    cd_values = []
    all_first_point_diffs = []
    all_last_point_diffs = []
    all_first_point_fit_diffs = []
    all_last_point_fit_diffs = []
    all_first_point_fit_mean_diffs = []
    all_last_point_fit_mean_diffs = []
    all_contour_square_diffs_first = []
    all_contour_square_diffs_last = []
    all_contour_square_diff_means = []

    straight_lines = extract_straight_lines(layout)
    show_image = cv2.cvtColor(layout, cv2.COLOR_GRAY2BGR)

    for contour in straight_lines:
        x, y, w, h = cv2.boundingRect(contour)
        short_axis = 'vertical' if w > h else 'horizontal'

        first_point_values = []
        last_point_values = []

        if short_axis == 'vertical':
            left, right = x, x + w
            start_x = (left + right) // 2

            mask = np.zeros_like(layout)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mask1 = np.zeros_like(layout)
            cv2.rectangle(mask1, (x, y), (x + w, y + h), 255, -1)

            # 向右移动
            x1 = start_x + direction
            while left <= x1 <= right and calculate_iou(mask[y:y + h, x1:x1 + 1],
                                                        mask1[y:y + h, x1:x1 + 1]) >= iou_threshold:
                x1 += direction
            x1 = min(x1, right)

            # 向左移动
            x2 = start_x - direction
            while left <= x2 <= right and calculate_iou(mask[y:y + h, x2:x2 + 1],
                                                        mask1[y:y + h, x2:x2 + 1]) >= iou_threshold:
                x2 -= direction
            x2 = max(x2, left)

            x_list.append([x2, x1])
            cv2.rectangle(show_image, (x2, y), (x1, y + h), (0, 0, 255), 2)

            contour_cd_values = []
            for col in range(x2, x1):
                column = mask[y:y + h, col:col + 1]
                line_length = np.sum(column > 0)
                contour_cd_values.append(line_length)

                nonzero_indices = np.where(column > 0)[0]
                if nonzero_indices.size > 0:
                    first_point = nonzero_indices[0]
                    last_point = nonzero_indices[-1]
                    first_point_values.append(first_point)
                    last_point_values.append(last_point)

            cd_values.append(contour_cd_values)
        else:
            top, bottom = y, y + h
            start_y = (top + bottom) // 2

            mask = np.zeros_like(layout)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mask1 = np.zeros_like(layout)
            cv2.rectangle(mask1, (x, y), (x + w, y + h), 255, -1)

            # 向下移动
            y1 = start_y + direction
            while top <= y1 <= bottom and calculate_iou(mask[y1:y1 + 1, x:x + w],
                                                        mask1[y1:y1 + 1, x:x + w]) >= iou_threshold:
                y1 += direction
            y1 = min(y1, bottom)

            # 向上移动
            y2 = start_y - direction
            while top <= y2 <= bottom and calculate_iou(mask[y2:y2 + 1, x:x + w],
                                                        mask1[y2:y2 + 1, x:x + w]) >= iou_threshold:
                y2 -= direction
            y2 = max(y2, top)

            x_list.append([y2, y1])
            cv2.rectangle(show_image, (x, y2), (x + w, y1), (0, 0, 255), 2)

            contour_cd_values = []
            for row in range(y2, y1):
                row_data = mask[row:row + 1, x:x + w]
                line_length = np.sum(row_data > 0)
                contour_cd_values.append(line_length)

                nonzero_indices = np.where(row_data > 0)[1]
                if nonzero_indices.size > 0:
                    first_point = nonzero_indices[0] + x
                    last_point = nonzero_indices[-1] + x
                    first_point_values.append(first_point)
                    last_point_values.append(last_point)

            cd_values.append(contour_cd_values)

        # 计算差异值
        if first_point_values and last_point_values:
            first_point_diff = max(first_point_values) - min(first_point_values)
            last_point_diff = max(last_point_values) - min(last_point_values)
        else:
            first_point_diff = 0
            last_point_diff = 0

        all_first_point_diffs.append(first_point_diff)
        all_last_point_diffs.append(last_point_diff)

        # 线性拟合
        def calculate_fit_diffs(values):
            if len(values) > 1:
                indices = np.arange(len(values))
                slope, intercept, _, _, _ = linregress(indices, values)
                fit_line = slope * indices + intercept
                fit_diffs = values - fit_line
                fit_mean_diff = np.mean(np.abs(fit_diffs))
            else:
                fit_diffs = np.array([0])
                fit_mean_diff = 0
            return fit_diffs, fit_mean_diff

        first_point_fit_diffs, first_point_fit_mean_diff = calculate_fit_diffs(first_point_values)
        last_point_fit_diffs, last_point_fit_mean_diff = calculate_fit_diffs(last_point_values)

        all_first_point_fit_diffs.append(first_point_fit_diffs)
        all_last_point_fit_diffs.append(last_point_fit_diffs)
        all_first_point_fit_mean_diffs.append(first_point_fit_mean_diff)
        all_last_point_fit_mean_diffs.append(last_point_fit_mean_diff)

        # 计算上下两条线的平方差
        def calculate_square_diffs(fit_diffs, values):
            if len(values) > 2:
                return np.sqrt(np.sum(fit_diffs ** 2) / (len(values) - 2))
            return np.sqrt(np.sum(fit_diffs ** 2) / len(values))

        square_diffs_first = calculate_square_diffs(first_point_fit_diffs, first_point_values)
        square_diff_last = calculate_square_diffs(last_point_fit_diffs, last_point_values)
        square_diff_mean = (square_diffs_first + square_diff_last) / 2

        all_contour_square_diffs_first.append(square_diffs_first)
        all_contour_square_diffs_last.append(square_diff_last)
        all_contour_square_diff_means.append(square_diff_mean)

    return x_list, show_image, cd_values, all_first_point_diffs, all_last_point_diffs, all_first_point_fit_diffs, all_last_point_fit_diffs, all_first_point_fit_mean_diffs, all_last_point_fit_mean_diffs, all_contour_square_diffs_first, all_contour_square_diffs_last, all_contour_square_diff_means


def process_images(image_folder, output_folder):
    """
    处理一系列图像。
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.bmp'))]
    all_cd_values = []
    all_diffs = []
    all_first_point_fit_diffs_list = []
    all_last_point_fit_diffs_list = []
    all_first_point_fit_mean_diffs_list = []
    all_last_point_fit_mean_diffs_list = []
    all_contour_square_diffs_first_list = []
    all_contour_square_diffs_last_list = []
    all_contour_square_diff_means_list = []

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        layout = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        results = measure_cd_values(layout)
        x_list, show_image, cd_values, first_point_diffs, last_point_diffs, first_point_fit_diffs, last_point_fit_diffs, first_point_fit_mean_diffs, last_point_fit_mean_diffs, contour_square_diffs_first_list, contour_square_diffs_last_list, contour_square_diff_means_list = results
        all_diffs.append((image_file, first_point_diffs, last_point_diffs))
        all_first_point_fit_diffs_list.append(first_point_fit_diffs)
        all_last_point_fit_diffs_list.append(last_point_fit_diffs)
        all_first_point_fit_mean_diffs_list.append(first_point_fit_mean_diffs)
        all_last_point_fit_mean_diffs_list.append(last_point_fit_mean_diffs)
        all_contour_square_diffs_first_list.append(contour_square_diffs_first_list)
        all_contour_square_diffs_last_list.append(contour_square_diffs_last_list)
        all_contour_square_diff_means_list.append(contour_square_diff_means_list)

        all_cd_values.append(cd_values)
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, show_image)
        print(f"Processed {image_file}")

    # 保存差异值到文件
    def save_differences(file_path, data, header):
        with open(file_path, 'w') as f:
            for image_file, *values in data:
                f.write(f"Image: {image_file}\n")
                for i, value in enumerate(zip(*values)):
                    f.write(f"Contour {i + 1}: {header.format(*value)}\n")
                f.write("\n")

    diff_txt_path = os.path.join(output_folder, 'all_differences.txt')
    save_differences(diff_txt_path, all_diffs, "First point diff: {}, Last point diff: {}")

    fit_diff_txt_path = os.path.join(output_folder, 'all_fit_differences.txt')
    fit_data = zip(image_files, all_first_point_fit_diffs_list, all_last_point_fit_diffs_list)
    save_differences(fit_diff_txt_path, fit_data, "First point fit diff: {}, Last point fit diff: {}")

    fit_mean_diff_txt_path = os.path.join(output_folder, 'all_fit_mean_differences.txt')
    fit_mean_data = zip(image_files, all_first_point_fit_mean_diffs_list, all_last_point_fit_mean_diffs_list)
    save_differences(fit_mean_diff_txt_path, fit_mean_data,
                     "First point fit mean diff: {}, Last point fit mean diff: {}")

    square_diff_txt_path = os.path.join(output_folder, 'all_square_differences.txt')
    square_data = zip(image_files, all_contour_square_diffs_first_list, all_contour_square_diffs_last_list,
                      all_contour_square_diff_means_list)
    save_differences(square_diff_txt_path, square_data,
                     "Square Diff First: {}, Square Diff last: {}, Square Diff Mean: {}")

    return all_cd_values


if __name__ == '__main__':
    image_folder = 'seg_3_epoch3/seg_3_epoch3'
    output_folder = 'self_evaluation_edgepoint'
    all_cd_values = process_images(image_folder, output_folder)

