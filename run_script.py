import torch
import torchaudio
import os

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from util.visualizer import Visualizer
from util.spectro_img import compute_visuals
from util.util import compute_matrics

# Initilize the setup
opt = TrainOptions().parse()
visualizer = Visualizer(opt)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
model_path = os.path.join(opt.checkpoints_dir, opt.name, 'model_scripted.pt')
model = torch.jit.load(model_path)
print('#audio segments = %d' % dataset_size)

# Forward pass
spectro_mag = []
spectro_pha = []
norm_params = []
audio = []
model.eval()
with torch.no_grad():
    for i, data in enumerate(dataset):
        sr_spectro, sr_audio, lr_pha, norm_param, lr_spectro = model.inference(
            data['label'])
        print(sr_spectro.size())
        spectro_mag.append(sr_spectro)
        spectro_pha.append(lr_pha)
        norm_params.append(norm_param)
        if opt.gen_overlap == 0:
            audio.append(sr_audio)
        else:
            audio.append(sr_audio[..., opt.gen_overlap//2:-opt.gen_overlap//2])

# Concatenate the audio
audio = torch.cat(audio, dim=0).view(1, -1)
audio_len = data_loader.dataset.raw_audio.size(-1)

# Compute metrics
_mse, _snr_sr, _snr_lr, _ssnr_sr, _ssnr_lr, _pesq, _lsd = compute_matrics(
    data_loader.dataset.raw_audio, data_loader.dataset.lr_audio[..., :audio_len], audio[..., :audio_len], opt)
print('MSE: %.4f' % _mse)
print('SNR_SR: %.4f' % _snr_sr)
print('SNR_LR: %.4f' % _snr_lr)
#print('SSNR_SR: %.4f' % _ssnr_sr)
#print('SSNR_LR: %.4f' % _ssnr_lr)
#print('PESQ: %.4f' % _pesq)
print('LSD: %.4f' % _lsd)
with open(os.path.join(opt.checkpoints_dir, opt.name, 'metric.txt'), 'w') as f:
    f.write('MSE,SNR_SR,LSD\n')
    f.write('%f,%f,%f' % (_mse, _snr_sr, _lsd))

sr_path = os.path.join(opt.checkpoints_dir, opt.name, 'sr_audio.wav')
lr_path = os.path.join(opt.checkpoints_dir, opt.name, 'lr_audio.wav')
hr_path = os.path.join(opt.checkpoints_dir, opt.name, 'hr_audio.wav')
torchaudio.save(sr_path, audio.cpu().to(torch.float32), opt.hr_sampling_rate)
torchaudio.save(lr_path, data_loader.dataset.lr_audio.cpu(),
                opt.hr_sampling_rate)
torchaudio.save(hr_path, data_loader.dataset.raw_audio.cpu(),
                data_loader.dataset.in_sampling_rate)
