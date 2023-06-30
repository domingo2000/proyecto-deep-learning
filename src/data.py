import os
import pandas as pd
from torch.utils.data import Dataset


class SCANDataset(Dataset):
    def __init__(
        self, path, transform=None, target_transform=None, label_transform=None
    ):
        with open(path) as file:
            instances = file.readlines()
            instances = [instance.replace("IN: ", "") for instance in instances]
            instances = [instance.split("OUT: ") for instance in instances]
            instances = [
                (instance[0].strip(), instance[1].strip()) for instance in instances
            ]

        self.instances = instances
        self.transform = transform
        self.target_transform = target_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        x_i = instance[0]
        y_i = instance[1]
        if self.label_transform:
            y_label = self.label_transform(y_i)
        if self.transform:
            x_i = self.transform(x_i)
        if self.target_transform:
            y_i = self.target_transform(y_i)

        return x_i, y_i, y_label
