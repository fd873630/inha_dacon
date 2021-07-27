import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import math
import pandas as pd
import random
import math

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


file_list = [[] for i in range(42711)]

with open("/home/jhjeong/jiho_deep/inha_dacon/inha_data/ID_List.txt") as f:
    files = f.read().splitlines()

for i, fi in enumerate(files):
    fi = fi.split()

    label = int(fi[0]) - 44165
    txt = fi[1]

    file_list[label].append((txt, fi[0]))

train = []
test = []

for i in file_list:
    
    random.shuffle(i)
    test_len = math.floor(len(i)*0.1)
    
    train += i[:-test_len]
    test += i[-test_len:]

with open("./train.txt", "w") as f:
    f.write("")

with open("./test.txt", "w") as f:
    f.write("")

for num, i in enumerate(train):
    
    if num % 1000 == 0:
        print(num)

    txt, label = i

    with open("./train.txt", "a") as f:
        f.write(str(label))
        f.write(" ")
        f.write(str(txt))
        f.write("\n")

for num, i in enumerate(test):
    
    if num % 1000 == 0:
        print(num)
    
    txt, label = i

    with open("./test.txt", "a") as f:
        f.write(str(label))
        f.write(" ")
        f.write(str(txt))
        f.write("\n")