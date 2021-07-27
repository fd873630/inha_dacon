import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import math
import pandas as pd

from PIL import Image

import cv2

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
import torchvision
from torchvision import datasets, models, transforms

data_transforms = {
    'train': transforms.Compose([       
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class MS1MDataset(Dataset):
    
    def __init__(self,split, id_list_path):

        self.file_list = id_list_path
        self.images = []
        self.labels = []
        self.transformer = data_transforms['train']
        
        self.data_path = id_list_path.split(split)[0]

        with open(self.file_list) as f:
            files = f.read().splitlines()

        for i, fi in enumerate(files):
            fi = fi.split()

            image = fi[1] 
            image = self.data_path + image
            label = int(fi[0]) - 44165 # min 값
         
            self.images.append(image)
            self.labels.append(label)
            
    def __getitem__(self, index):

        img = Image.open(self.images[index])
        img = self.transformer(img)

        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)