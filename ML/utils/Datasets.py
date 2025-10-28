from functools import cache
import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from torchvision.transforms import functional as F


labels_names =   [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
  ]

class AffectNet_dataset(Dataset):
    def __init__(self, root="ML/datasets/affectnet-yolo-format", is_test=False, transform=None, cache_to_ram=False):
        split = "valid" if is_test else "train"
        self.root = os.path.join(root, split)
        self.img_dir = os.path.join(self.root, "images")
        self.lbl_dir = os.path.join(self.root, "labels")
        self.image_names = [n for n in os.listdir(self.img_dir) if n.lower().endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, name)
        lbl_path = os.path.join(self.lbl_dir, name.rsplit('.', 1)[0] + ".txt")
        
        with open(lbl_path, "r") as f:
            label = int(f.readline()[0])

        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
