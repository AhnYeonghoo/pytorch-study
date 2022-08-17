import torch
from torch import hub

resnet18_model = hub.load('pytorch/visionL:master',
                          'resnet18',
                          pretrained=True)