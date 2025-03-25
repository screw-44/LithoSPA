import torch
import torch.utils.data as data
import random
import json

from torchvision.transforms import Compose

class Normalize:
    def __call__(self, x, y):
        return x / 255.0, y / 30.0

class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x, y):
        if random.random() < self.p:
            return x, y
        return torch.flip(x, dims=[-1]), torch.flip(y, dims=[-1])

class EdgeScenario:
    def __init__(self, p=0.2):
        self.p = p
        self.max_range = 0.6

    def __call__(self, x, y:torch.Tensor):
        if random.random() < self.p:
            return x, y
        # argument x
        mask_direction = random.randint(0, 1)
        mask_size = max(1, int(len(x) * random.random() * self.max_range)) # mask的最大大小为1～长度的0.6
        mask_index = list(range(mask_size))
        if mask_direction: # 从后往前的mask情况, mask移动到后半部份
            mask_index = [_ + len(x) - mask_size for _ in mask_index]
        x[mask_index] = -1
        # argument y
        # 根据mask的前后，调整y
        if mask_direction: # 从后往前
            y.clamp_(max = mask_index[0] / 30)
        else: # 从前往后
            y.clamp_(min = mask_index[-1] / 30)

        return x, y

class RandomGaussianNoise:
    def __init__(self, max_mean=0.1, std=0.05):
        self.max_mean, self.std = max_mean, std

    def __call__(self, x, y):
        return x + torch.randn(x.size()) * self.std + random.random()*self.max_mean, y



class Dataset(data.Dataset):

    def __init__(self, is_val=False):
        super(Dataset, self).__init__()
        file_path = "train_onnx.json" if not is_val else "val_onnx.json"
        with open(file_path, "r") as f:
            dataset = json.load(f)
        self.is_val = is_val

        self.X, self.Y = [], []
        for _ in dataset["edges"]:
            self.X.append(_["brightness_values"])
            self.Y.append([_["inner_edge_index"], _["center_edge_index"], _["outer_edge_index"],])

        self.transforms = Compose([
            Normalize,
            RandomFlip,
            RandomGaussianNoise,
            EdgeScenario,
        ])

        self.t_normalize = Normalize()
        self.t_randomFlip = RandomFlip()
        self.t_randomGaussianNoise = RandomGaussianNoise()
        self.t_edgeScenario = EdgeScenario()

        self.device = torch.device("mps:0" if torch.mps.is_available() else "cpu")

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        x, y = self.t_normalize(x, y)
        if self.is_val:
            return x.to(self.device), y.to(self.device)
        x, y = self.t_randomFlip(x, y)
        x, y = self.t_randomGaussianNoise(x, y)
        # x, y = self.t_edgeScenario(x, y)
        return x.to(self.device), y.to(self.device)

    def __len__(self):
        return len(self.X)