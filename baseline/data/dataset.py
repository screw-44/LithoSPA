import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import os

from torchvision.datasets.folder import IMG_EXTENSIONS

def get_transform():
    transform_list = [transforms.Resize([256, 256]), transforms.ToTensor()]
    return transforms.Compose(transform_list)

def is_image_file(filename): return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(_dir):
    images_paths = []
    assert os.path.isdir(_dir)

    for root, folder_name, filenames in sorted(os.walk(_dir)):
        for filename in filenames:
            if is_image_file(filename):
                images_paths.append(os.path.join(root, filename))

    return images_paths

class AlignedDataset(data.Dataset):

    def __init__(self, opt):
        super(AlignedDataset, self).__init__()

        self.opt = opt
        # layout path
        self.layout_paths = sorted(make_dataset(opt.layout_image_dir))
        # sem path
        self.sem_paths = sorted(make_dataset(opt.sem_image_dir))
        # seg gt path
        self.gt_paths = sorted(make_dataset(opt.gt_dir))


    def __getitem__(self, index):
        transform = get_transform()
        # layout
        layout_path = self.layout_paths[index]
        layout = Image.open(layout_path).convert('L')
        layout_tensor = transform(layout)

        # sem
        sem_path = self.sem_paths[index]
        sem_image = Image.open(sem_path).convert('L')
        sem_tensor = transform(sem_image)

        # segmentation gt
        gt_path = self.gt_paths[index]
        gt_image = Image.open(gt_path).convert('L')
        gt_image = transform(gt_image)

        input_dict = {'layout': layout_tensor, 'sem': sem_tensor, 'gt': gt_image.float(), 'path': layout_path}
        return input_dict

    def __len__(self):
        return len(self.layout_paths) // self.opt.batch_size * self.opt.batch_size

    @property
    def name(self):
        return 'AlignedDataset'

