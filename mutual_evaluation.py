import os
import torch
import torch.nn.functional as func
import numpy as np
import cv2

# 常量定义
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REALTYPE = torch.float32
EPE_CONSTRAINT = 1  # 根据图像分辨率调整

class Basic:
    def __init__(self, thresh=0.5, device=DEVICE):
        self._thresh = thresh
        self._device = device

    def run(self, mask, target):
        """
        计算 L2 损失和 PVBand
        :param mask: 输入掩码图像 (H, W)
        :param target: 目标图像 (H, W)
        :return: L2 损失, PVBand
        """
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)

        # 强制四舍五入并二值化
        mask = torch.round(mask)
        target = torch.round(target)
        mask = (mask >= self._thresh).float()
        target = (target >= self._thresh).float()

        # 使用MSE计算 L2 损失
        l2loss = func.mse_loss(mask, target, reduction="sum")

        # 计算 PVBand（mask 和 target 的差异区域）意义就是相同为1，不同为0，并且相加起来
        pvband = torch.sum(mask != target)

        return l2loss.item(), pvband.item()





def boundaries(target):
    """
    提取目标图像的边界，并区分垂直边界和水平边界
    :param target: 目标图像 (H, W)
    :return: 垂直边界点对, 水平边界点对
    """
    # 使用形态学梯度检测边界
    kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=target.device)
    kernel = kernel[None, None, :, :]  # 增加 batch 和 channel 维度

    # 膨胀操作
    dilated = func.conv2d(target[None, None, :, :], kernel, padding=1)[0, 0]

    # 腐蚀操作（通过卷积实现）
    eroded = func.conv2d(target[None, None, :, :], kernel, padding=1)[0, 0]
    eroded = (eroded >= 4).float()  # 腐蚀条件：中心像素的邻域和 >= 4

    # 边界 = 膨胀 - 腐蚀
    boundary = (dilated - eroded).clamp(0, 1)

    # 提取垂直边界（左右边界）
    vertical_kernel = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float32, device=target.device)
    vertical_kernel = vertical_kernel[None, None, :, :]
    vertical = func.conv2d(boundary[None, None, :, :], vertical_kernel, padding=1)[0, 0]
    vertical = (vertical > 0).float()  # 提取垂直边界

    # 提取水平边界（上下边界）
    horizontal_kernel = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.float32, device=target.device)
    horizontal_kernel = horizontal_kernel[None, None, :, :]
    horizontal = func.conv2d(boundary[None, None, :, :], horizontal_kernel, padding=1)[0, 0]
    horizontal = (horizontal > 0).float()  # 提取水平边界

    # 找到垂直和水平边界点的坐标
    vsites = vertical.nonzero(as_tuple=False).float()
    hsites = horizontal.nonzero(as_tuple=False).float()

    return vsites, hsites


def check(image, sample, target, direction):
    """
    检查边界点的 EPE 误差
    :param image: 输入掩码图像 (H, W)
    :param sample: 边界点 (N, 2)
    :param target: 目标图像 (H, W)
    :param direction: 检查方向 ('v' 或 'h')
    :return: 内部误差点, 外部误差点
    """
    inner = []
    outer = []

    for point in sample:
        x, y = int(point[0]), int(point[1])

        # 检查坐标是否有效
        if x < 0 or x >= image.shape[0] or y < 0 or y >= image.shape[1]:
            continue

        if direction == 'v':  # 垂直方向
            if y + 1 >= target.shape[1] or y - 1 < 0:
                continue
            if target[x, y + 1] == 1 and target[x, y - 1] == 0:  # 左边界
                inner_point = (x, y + EPE_CONSTRAINT)
                outer_point = (x, y - EPE_CONSTRAINT)
            elif target[x, y + 1] == 0 and target[x, y - 1] == 1:  # 右边界
                inner_point = (x, y - EPE_CONSTRAINT)
                outer_point = (x, y + EPE_CONSTRAINT)
            else:
                continue  # 跳过非单侧边界点

        elif direction == 'h':  # 水平方向
            if x + 1 >= target.shape[0] or x - 1 < 0:
                continue
            if target[x + 1, y] == 1 and target[x - 1, y] == 0:  # 上边界
                inner_point = (x + EPE_CONSTRAINT, y)
                outer_point = (x - EPE_CONSTRAINT, y)
            elif target[x + 1, y] == 0 and target[x - 1, y] == 1:  # 下边界
                inner_point = (x - EPE_CONSTRAINT, y)
                outer_point = (x + EPE_CONSTRAINT, y)
            else:
                continue  # 跳过非单侧边界点

        # 检查 inner_point 和 outer_point 是否有效
        def is_valid(p):
            return 0 <= p[0] < image.shape[0] and 0 <= p[1] < image.shape[1]

        # 检查内部误差
        if is_valid(inner_point):
            if image[inner_point[0], inner_point[1]] == 0:
                inner.append([inner_point[0], inner_point[1]])

        # 检查外部误差
        if is_valid(outer_point):
            if image[outer_point[0], outer_point[1]] == 1:
                outer.append([outer_point[0], outer_point[1]])

    # 转换为 Tensor
    inner = torch.tensor(inner, dtype=torch.float32, device=image.device) if inner else torch.tensor([], dtype=torch.float32, device=image.device)
    outer = torch.tensor(outer, dtype=torch.float32, device=image.device) if outer else torch.tensor([], dtype=torch.float32, device=image.device)
    return inner, outer

def calculate_shots(mask):
    """
    计算掩码中的矩形数量（shot）
    :param mask: 输入掩码图像 (H, W)
    :return: 矩形数量
    """
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=REALTYPE, device=DEVICE)

    # 将掩码转换为二值图像
    mask = (mask >= 0.5).float()

    # 将掩码转换为 NumPy 数组
    mask_np = mask.cpu().numpy().astype(np.uint8)

    # 使用 OpenCV 的 connectedComponentsWithStats 检测连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

    # 统计矩形数量
    shot_count = 0
    for label in range(1, num_labels):  # 跳过背景（label=0）
        # 提取当前连通区域的边界框
        x, y, w, h, area = stats[label]
        shot_count += 1  # 每个连通区域至少是一个矩形

        # 如果需要进一步分解连通区域为多个矩形，可以在此处添加逻辑
        # 例如，使用轮廓检测和矩形拟合

    return shot_count

def epecheck(mask, target, vposes, hposes):
    """
    计算 EPE（边缘放置误差）
    :param mask: 输入掩码图像 (H, W)
    :param target: 目标图像 (H, W)
    :param vposes: 垂直边界点对 (N_v, 2)
    :param hposes: 水平边界点对 (N_h, 2)
    :return: EPE 内部误差, EPE 外部误差, 误差图
    """
    inner = 0
    outer = 0
    vioMap = torch.zeros_like(target)

    # 检查垂直边界
    if vposes.numel() > 0:
        for point in vposes:
            v_in_site, v_out_site = check(mask, point.unsqueeze(0), target, 'v')
            inner += v_in_site.shape[0]
            outer += v_out_site.shape[0]
            if v_in_site.numel() > 0:
                vioMap[v_in_site[:, 0].long(), v_in_site[:, 1].long()] = 1
            if v_out_site.numel() > 0:
                vioMap[v_out_site[:, 0].long(), v_out_site[:, 1].long()] = 1

    # 检查水平边界
    if hposes.numel() > 0:
        for point in hposes:
            h_in_site, h_out_site = check(mask, point.unsqueeze(0), target, 'h')
            inner += h_in_site.shape[0]
            outer += h_out_site.shape[0]
            if h_in_site.numel() > 0:
                vioMap[h_in_site[:, 0].long(), h_in_site[:, 1].long()] = 1
            if h_out_site.numel() > 0:
                vioMap[h_out_site[:, 0].long(), h_out_site[:, 1].long()] = 1

    return inner, outer, vioMap


class EPEChecker:
    def __init__(self, thresh=0.5, device=DEVICE):
        self._thresh = thresh
        self._device = device

    def run(self, mask, target):
        """
        计算 EPE
        :param mask: 输入掩码图像 (H, W)
        :param target: 目标图像 (H, W)
        :return: EPE 内部误差, EPE 外部误差
        """
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)

        # 强制四舍五入并二值化
        mask = torch.round(mask)
        target = torch.round(target)
        mask = (mask >= self._thresh).float()
        target = (target >= self._thresh).float()

        # 提取边界
        vposes, hposes = boundaries(target)

        # 计算 EPE
        epeIn, epeOut, _ = epecheck(mask, target, vposes, hposes)
        return epeIn, epeOut

def calculate_iou(mask, target):
    """
    计算 IoU（交并比）
    :param mask: 输入掩码图像 (H, W)
    :param target: 目标图像 (H, W)
    :return: IoU 值
    """
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=REALTYPE, device=DEVICE)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=REALTYPE, device=DEVICE)

    # 二值化
    mask = (mask >= 0.5).float()
    target = (target >= 0.5).float()

    # 计算交集和并集
    intersection = torch.sum(mask * target)  # 交集
    union = torch.sum(mask) + torch.sum(target) - intersection  # 并集

    # 计算 IoU
    iou = intersection / union if union > 0 else 0.0

    return iou.item()

def process_folder(folder_path, output_file):
    """
    处理文件夹中的所有图像文件，并将结果保存到文本文件中
    :param folder_path: 文件夹路径
    :param output_file: 输出文件路径
    """
    # 初始化工具
    test = Basic()
    epeCheck = EPEChecker()

    # 打开输出文件
    with open(output_file, "w") as f:
        # 写入表头
        f.write("File,L2,PVBand,EPE,Shot,IoU\n")

        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                # 读取图像
                mask_path = os.path.join(folder_path, filename)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
                target = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255  # 假设目标图像与掩码图像相同

                # 计算指标
                l2, pvb = test.run(mask, target)
                epeIn, epeOut = epeCheck.run(mask, target)
                epe = epeIn + epeOut
                shot = calculate_shots(mask)
                iou = calculate_iou(mask, target)

                # 写入结果
                f.write(f"{filename},{l2:.0f},{pvb:.0f},{epe:.0f},{shot:.0f},{iou:.4f}\n")

if __name__ == "__main__":
    # 示例文件夹路径和输出文件路径
    folder_path = "我们的输入文件"  # 替换为您的文件夹路径
    output_file = "输出结果文件，保存为.txt"  # 输出文件路径

    # 处理文件夹并保存结果
    process_folder(folder_path, output_file)