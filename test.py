# Basic testing procedure taken from https://github.com/suriyadeepan/torchtest/blob/master/examples.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src import models
from helper import get_models

# set gpu for avoiding possible errors in torchtest
dev='cuda:0'
torch.device(dev)

# get the model names
model_names = get_models(models)

import torchtest as tt
inputs = torch.rand(2, 3, 224, 224).to(dev)
targets = torch.FloatTensor(2).uniform_(0, 1000).long().to(dev)
# torch.randint(0, 2, (2, 1000,))
batch = [inputs, targets]
model = models.bl_resnet50().to(dev)

# what are the variables?
print('Our list of parameters', [ np[0] for np in model.named_parameters() ])

# do they change after a training step?
#  let's run a train step and see
tt.assert_vars_change(
    model=model, 
    loss_fn=F.cross_entropy,
    optim=torch.optim.Adam(model.parameters()),
    batch=batch)

print("All the parameters are getting modified.")
