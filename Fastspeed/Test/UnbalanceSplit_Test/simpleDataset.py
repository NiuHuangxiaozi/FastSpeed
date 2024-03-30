from torch.utils.data import  Dataset
import numpy as np
import torch


class DummyDataset(Dataset):
    def __init__(self):
        xy =list(range(50))
        self.x_data = torch.LongTensor(xy).reshape(-1,1)
        self.y_data = torch.LongTensor(xy)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len