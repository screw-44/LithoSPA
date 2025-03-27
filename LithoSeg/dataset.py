import os
import glob
import torch.utils.data as data
import cv2
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class FinetuneDataset(data.Dataset):
    def __init__(self, label_dir = "./dataset/train_seg_3(epoch3)"):
        super(FinetuneDataset, self).__init__()
        self.label_dirs = glob.glob(f"{label_dir}/*")
        self.adi_dirs = []
        self.masks_dirs = []
        for _ in self.label_dirs:
            filename = os.path.basename(_).split(".")[0]
            self.adi_dirs.append(f"../sjw/layout2adi0907/train/ADI/{filename}.bmp")
            self.masks_dirs.append(f"../sjw/layout2adi0907/train/layout/{filename}.png")
        self.dataset_size = len(self.label_dirs)
        self.input_transforms = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            ]
        )
        self.label_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]
        )

        # preprocess the masks into bbox points
        self.box_prompt = []
        for mask_dir in self.masks_dirs:
            mask_image = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
            mask_image = cv2.bitwise_not(mask_image)
            mask_image = cv2.resize(mask_image, (1024, 1024))

            mask_image[0:15, :] = 0 # 把边缘set为0
            mask_image[-15:, :] = 0
            mask_image[:, 0:15] = 0
            mask_image[:, -15:] = 0

            contours, _ = cv2.findContours(mask_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(mask_image, (x, y), (x+w, y+h), (255, 0, 0), 4)
                boxes.append([x, y, x + w, y + h])  # 左上和右下
            # plt.imshow(mask_image)
            # plt.show()

            self.box_prompt.append(boxes)



    def __getitem__(self, index):
        label = self.label_transforms(Image.open(self.label_dirs[index]).convert('L'))
        adi = self.input_transforms(Image.open(self.adi_dirs[index]).convert('RGB'))
        box_prompt = torch.tensor(self.box_prompt[index])
        #  3xHxW format 检查是否为这个输入格式
        return {"label": label, "adi": adi, "box_prompt": box_prompt, "index_dir": self.label_dirs[index]}

    def __len__(self):
        return self.dataset_size
