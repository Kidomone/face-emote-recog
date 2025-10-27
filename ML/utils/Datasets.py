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
    def __init__(self, path_to_dict="ML/datasets/affectnet-yolo-format", is_test=False, transform=None, cache_to_ram=True):
        self.images = []
        self.labels = []
        self.transform = transform
        self.cache_to_ram = cache_to_ram

        if is_test:
            path_to_dict = os.path.join(path_to_dict, 'valid')
        else:
            path_to_dict = os.path.join(path_to_dict, 'train')

        path_to_imgs = os.path.join(path_to_dict, 'images')
        path_to_lbls = os.path.join(path_to_dict, 'labels')

        for image_name in os.listdir(path_to_imgs):
            try:
                image_path = os.path.join(path_to_imgs, image_name)

                if cache_to_ram:
                    img = Image.open(image_path).convert("L")
                    
                    img = F.crop(img, 6, 14, 84, 70)                                     # TODO Check

                    img_tensor = torch.from_numpy(np.array(img))
                    self.images.append(img_tensor)
                else:
                    self.images.append(image_path)

                label_path = os.path.join(path_to_lbls, image_name[:-3])
                with open(label_path+'txt', 'r') as file:
                    line = file.readline()
                    digit = int(line[0])
                self.labels.append(digit)

            except Exception as e:
                print(f"Error in {self.__class__.__name__} - {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.cache_to_ram:
            img = Image.fromarray(img.numpy())
        else:
            img = Image.open(img).convert("L")

        img = F.crop(img, 6, 14, 84, 70)                                                 # TODO Check

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label
