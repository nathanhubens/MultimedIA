import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from fastai.vision import *
  
  
def get_data(path, bs,size, tfms, device):
    src = ImageList.from_folder(path).split_by_folder(train='train', valid='test')
    data = (src.label_from_folder()
            .transform(get_transforms(xtra_tfms=tfms), size=size)
            .databunch(bs=bs, device=device).normalize())
    return data

class MLPModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, output_size)
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(-1, self.input_size)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        return out


class CNNModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 3,1, 1)
        self.fc1 = nn.Linear(32*8*8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32*8*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def count_parameters(model):
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {n_params:,}')