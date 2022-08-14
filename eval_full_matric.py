import torchaudio
from data.data_loader import CreateDataLoader
from util.util import compute_matrics
from options.train_options import TrainOptions
from models.models import create_model
import math
import os
import torch
import csv


def lcm(a, b): return abs(a * b)/math.gcd(a, b) if a and b else 0


def get_file_list(file_path):
    root, csv_file = os.path.split(file_path)
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        file_list = [os.path.join(root, item) for sublist in list(
            csv_reader) for item in sublist]
    print(len(file_list))
    return file_list


# import debugpy
# debugpy.listen(("localhost", 5678))
# debugpy.wait_for_client()
# os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['NCCL_P2P_DISABLE'] = '1'
# Get the training options
opt = TrainOptions().parse()
opt.phase = 'test'
# Set the seed
torch.manual_seed(opt.seed)
# Set the path for save the trainning losses
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'eval.csv')

file_list = get_file_list(opt.dataroot)
dataset_size = len(file_list)
if opt.fp16:
    from torch.cuda.amp import autocast as autocast
# Create the model
model = create_model(opt)
model = model.eval()

# Set frequency for displaying information and saving
opt.print_freq = lcm(opt.print_freq, opt.batchSize)
start_epoch = 1

for epoch in range(start_epoch, opt.niter+1):
    for i, audio_file in enumerate(file_list):
        opt.dataroot = audio_file
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        err = []
        snr = []
        snr_seg = []
        pesq = []
        lsd = []
        ############## Forward Pass ######################
        audio = []
        for j, data in enumerate(dataset):
            lr_audio = data['LR_audio'].cuda()
            with torch.no_grad():
                if opt.fp16:
                    with autocast():
                        sr_spectro, sr_audio, lr_pha, norm_param, lr_spectro = model.inference(
                            lr_audio)
                else:
                    sr_spectro, sr_audio, lr_pha, norm_param, lr_spectro = model.inference(
                        lr_audio)
            audio.append(sr_audio)

        # Concatenate the audio
        if opt.gen_overlap > 0:
            stride = opt.segment_length-opt.gen_overlap
            from torch.nn.functional import fold
            out_len = (len(data_loader)-1) * stride + opt.segment_length
            print(out_len)
            audio = torch.cat(audio, dim=0)
            audio[..., :opt.gen_overlap] *= 0.5
            audio[..., -opt.gen_overlap:] *= 0.5
            audio = audio.squeeze().transpose(-1, -2)
            audio = fold(audio, kernel_size=(1, opt.segment_length), stride=(
                1, stride), output_size=(1, out_len)).squeeze(0)
            sr_audio = audio[..., opt.gen_overlap//2:-opt.gen_overlap//2]
        else:
            sr_audio = torch.cat(audio, dim=0).view(1, -1)
        hr_audio = data_loader.dataset.raw_audio
        lr_audio = data_loader.dataset.lr_audio
        ############## Evaluation Pass ####################
        if opt.hr_sampling_rate != opt.sr_sampling_rate:
            hr_audio = torchaudio.functional.resample(
                hr_audio, data_loader.dataset.in_sampling_rate, opt.sr_sampling_rate).squeeze().cpu()
            lr_audio = torchaudio.functional.resample(
                lr_audio, opt.hr_sampling_rate, opt.sr_sampling_rate).squeeze().cpu()
            sr_audio = torchaudio.functional.resample(
                sr_audio.cpu(), opt.hr_sampling_rate, opt.sr_sampling_rate).squeeze()

        print(hr_audio.size(1), sr_audio.size(1), lr_audio.size(1))
        length = min(hr_audio.size(1), sr_audio.size(1), lr_audio.size(1))

        _mse, _snr_sr, _, _ssnr_sr, _, _pesq, _lsd = compute_matrics(
            hr_audio[..., :length], lr_audio[..., :length], sr_audio[..., :length], opt)
        eval_result = {'file': audio_file, 'err': _mse, 'snr': _snr_sr,
                       'snr_seg': _ssnr_sr, 'pesq': _pesq, 'lsd': _lsd}
        with open(eval_path, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=eval_result.keys())
            if csv_file.tell() == 0:
                writer.writeheader()
            writer.writerow(eval_result)
        print('Evaluation:', eval_result)

        if i >= dataset_size:
            break
