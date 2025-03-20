from torch.utils.data import DataLoader, Subset
from baseline.data.dataset import AlignedDataset


class AlignedDatasetLoader:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = AlignedDataset(opt)
        self.dataset = Subset(self.dataset, list(range(50)))
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.opt.batch_size,
            shuffle=self.opt.shuffle,
            num_workers=self.opt.num_workers,
        )

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def name():
        return "AlignedDatasetLoader"

    def load_data(self):
        return self.dataloader
