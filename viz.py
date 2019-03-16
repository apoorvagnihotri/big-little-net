from src.models import *
import torch
import torchvision

from tensorboardX import SummaryWriter

model = bl_resnet50()

dummy_input = torch.rand(1, 3, 224, 224)

with SummaryWriter(comment='bl_resnet50') as w:
    model = bl_resnet50()
    w.add_graph(model, (dummy_input, ), verbose=True)
