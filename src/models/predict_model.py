import torch
from resnet import ResNet_18

model = ResNet_18(10)

input = torch.rand(1, 3 ,112 , 112)
model = ResNet_18(10)

x = model(input)