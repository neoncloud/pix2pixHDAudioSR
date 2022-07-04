import torch
import torchaudio
import os

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from util.util import compute_matrics

# Initilize the setup
opt = TrainOptions().parse()
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
stride = opt.segment_length-opt.gen_overlap
with torch.no_grad():
    for i, data in enumerate(dataset):
        sr_spectro, sr_audio, lr_pha, norm_param, lr_spectro = model.inference(
            data['label'])
        print(sr_spectro.size())
        spectro_mag.append(sr_spectro)
        spectro_pha.append(lr_pha)
        norm_params.append(norm_param)
        audio.append(sr_audio)

# Concatenate the audio
if opt.gen_overlap > 0:
    from torch.nn.functional import fold
    out_len = (dataset_size-1) * stride + opt.segment_length
    print(out_len)
    audio = torch.cat(audio,dim=0)
    audio[...,:opt.gen_overlap] *= 0.5
    audio[...,-opt.gen_overlap:] *= 0.5
    audio = audio.squeeze().transpose(-1,-2)
    print(audio.shape)
    audio = fold(audio, kernel_size=(1,opt.segment_length), stride=(1,stride), output_size=(1,out_len)).squeeze(0)
    audio = audio[...,opt.gen_overlap//2:-opt.gen_overlap//2]
    print(audio.shape)
else:
    audio = torch.cat(audio, dim=0).view(1, -1)
audio_len = data_loader.dataset.raw_audio.size(-1)

# Compute metrics
K = 16000
_mse, _snr_sr, _snr_lr, _ssnr_sr, _ssnr_lr, _pesq, _lsd = compute_matrics(
    data_loader.dataset.raw_audio[..., K:audio_len-K], data_loader.dataset.lr_audio[..., K:audio_len-K], audio[..., K:audio_len-K], opt)
print('MSE: %.4f' % _mse)
print('SNR_SR: %.4f' % _snr_sr)
print('SNR_LR: %.4f' % _snr_lr)
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
