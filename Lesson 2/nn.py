import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from fastai.vision import *
from IPython import display
  
  
def get_data(path, bs,size, tfms, device):
    src = ImageList.from_folder(path).split_by_folder(train='train', valid='test')
    data = (src.label_from_folder()
            .transform(get_transforms(xtra_tfms=tfms), size=size)
            .databunch(bs=bs, device=device).normalize())
    return data

def get_mnist_data(bs,size, tfms, device):
    path = untar_data(URLs.MNIST)
    src = ImageList.from_folder(path).split_by_folder(train='training', valid='testing')
    data = (src.label_from_folder()
            .transform((tfms,None), size=size)
            .databunch(bs=bs, device=device))
    return data


class MLPModel(nn.Module):
    def __init__(self, input_size, n_features, output_size):
        super().__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, int(n_features))
        self.linear2 = nn.Linear(int(n_features), int(n_features/2))
        self.linear3 = nn.Linear(int(n_features/2), int(n_features/4))
        self.linear4 = nn.Linear(int(n_features/4), int(n_features/8))
        self.linear5 = nn.Linear(int(n_features/8), output_size)
        
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
    def __init__(self, n_features, output_size):
        super().__init__()
        self.n_features = n_features
        self.conv1 = nn.Conv2d(3, int(n_features), 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(int(n_features), int(n_features*2), 3, 1, 1)
        self.conv3 = nn.Conv2d(int(n_features*2), int(n_features*4), 3,1, 1)
        self.fc1 = nn.Linear(int(n_features*4)*8*8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, (self.n_features*4)*8*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MNISTModel(nn.Module):
    def __init__(self, n_features, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, int(n_features), 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(int(n_features), int(n_features)*2, 3, 1, 1)
        self.conv3 = nn.Conv2d(int(n_features)*2, int(n_features)*4, 3,1, 1)
        self.fc1 = nn.Linear(int(n_features)*4*7*7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, int(n_features)*4*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def count_parameters(model):
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {n_params:,}')

def train(model, X,y):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for t in range(1000):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        score, predicted = torch.max(y_pred, 1)
        acc = (y == predicted).sum().float() / len(y)
        print("[EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f" % (t, loss.item(), acc))
        display.clear_output(wait=True)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()