import torch
import os

from options.train_options import TrainOptions
from models.models import create_model

# Initilize the setup
opt = TrainOptions().parse()
opt.isTrain = False
model = create_model(opt)
model_scripted = torch.jit.script(model)
torch.jit.save(model_scripted,os.path.join(opt.checkpoints_dir, opt.name, 'model_scripted.pt'))