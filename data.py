import torch
import pandas as pd
import cv2
from torch.utils.data import Dataset
import os

class CustomData(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotation = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        img_path = self.annotation.iloc[index, 0]
        img = cv2.imread(str(img_path))

        if self.transform:
            img = self.transform(img)

        y_label = torch.tensor(self.annotation.iloc[index, 3:6])

        return (img, y_label)



