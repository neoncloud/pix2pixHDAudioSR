import torchaudio
from data.data_loader import CreateDataLoader
from util.util import compute_matrics
from options.train_options import TrainOptions
from models.models import create_model
import math
import os
import time
import torch
import csv
import numpy as np
#from torch.autograd import Variable
#from prefetch_generator import BackgroundGenerator


def lcm(a, b): return abs(a * b)/math.gcd(a, b) if a and b else 0


# import debugpy
# debugpy.listen(("localhost", 5678))
# debugpy.wait_for_client()
# os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['NCCL_P2P_DISABLE'] = '1'
# Get the training options
opt = TrainOptions().parse()
# Set the seed
torch.manual_seed(opt.seed)
# Set the path for save the trainning losses
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'eval.csv')

# Create the data loader
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#evaluation data = %d' % dataset_size)
if opt.fp16:
    from torch.cuda.amp import autocast as autocast
# Create the model
model = create_model(opt)
model = model.eval()

# Set frequency for displaying information and saving
opt.print_freq = lcm(opt.print_freq, opt.batchSize)
start_epoch, epoch_iter = 1, 0
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
total_steps = (start_epoch-1) * dataset_size + epoch_iter
print_delta = total_steps % opt.print_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    err = []
    snr = []
    snr_seg = []
    pesq = []
    lsd = []
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        ############## Forward Pass ######################
        lr_audio = data['LR_audio'].cuda()
        hr_audio = data['HR_audio'].cuda()
        with torch.no_grad():
            if opt.fp16:
                with autocast():
                    sr_spectro, sr_audio, lr_pha, norm_param, lr_spectro = model.inference(
                        lr_audio)
            else:
                sr_spectro, sr_audio, lr_pha, norm_param, lr_spectro = model.inference(
                    lr_audio)

        ############## Evaluation Pass ####################
        if opt.hr_sampling_rate != opt.sr_sampling_rate:
            hr_audio = torchaudio.functional.resample(hr_audio,opt.hr_sampling_rate,opt.sr_sampling_rate).squeeze()
            lr_audio = torchaudio.functional.resample(lr_audio,opt.hr_sampling_rate,opt.sr_sampling_rate).squeeze()
            sr_audio = torchaudio.functional.resample(sr_audio,opt.hr_sampling_rate,opt.sr_sampling_rate).squeeze()
            
        _mse, _snr_sr, _snr_lr, _ssnr_sr, _ssnr_lr, _pesq, _lsd = compute_matrics(
            hr_audio, lr_audio.squeeze(), sr_audio.squeeze(), opt)
        err.append(_mse)
        snr.append((_snr_lr, _snr_sr))
        snr_seg.append((_ssnr_lr, _ssnr_sr))
        pesq.append(_pesq)
        lsd.append(_lsd)
        eval_result = {'err': np.mean(err), 'snr': np.mean(snr), 'snr_seg': np.mean(
            snr_seg), 'pesq': np.mean(pesq), 'lsd': np.mean(lsd)}
        with open(eval_path, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=eval_result.keys())
            if csv_file.tell() == 0:
                writer.writeheader()
            writer.writerow(eval_result)
        print('Evaluation:', eval_result)
        if epoch_iter >= dataset_size:
            break
