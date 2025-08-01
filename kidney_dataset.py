import os
import torch
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class YoloToFasterRCNNDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img)
        h, w, _ = img.shape

        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.strip().split())
                xmin = (x - bw/2) * w
                xmax = (x + bw/2) * w
                ymin = (y - bh/2) * h
                ymax = (y + bh/2) * h
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(cls) + 1)  # background = 0, so class_id starts from 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        return img_tensor, target

    def __len__(self):
        return len(self.image_files)
