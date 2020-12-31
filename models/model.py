import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import re
import sys
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import torchvision


class CustomDensenet(nn.Module):
    def __init__(self, hidden_size=256, num_classes=2, convert=False):
        super(CustomDensenet, self).__init__()
        dense = models.densenet161(pretrained=True)
        self.convert = convert
        self.densenet = nn.Sequential(*list(dense.children())[:-1])
        self.avg = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(2208, hidden_size)
        self.relu = nn.ReLU()
        self.dr1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.device = self.init_device()

    def init_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        h = self.densenet(x)
        h = self.avg(h)
        h_middle = h.view(len(h), -1)
        h = self.dr1(self.relu(self.fc1(h_middle)))
        y = self.fc2(h)
        
        return y, h_middle
