import streamlit as st
import torch
import torchaudio
import matplotlib.pyplot as plt
from os.path import splitext
from io import BytesIO
import numpy as np

st.title('Audio Upload')
uploaded_audio = st.file_uploader("Please upload a audio file", ['wav', 'mp3', 'flac', 'ogg'], False)
if uploaded_audio is not None:
    #metadata = torchaudio.info(uploaded_audio)
    #st.write('Audio Length:',metadata.num_frames)
    #cache_filename = './cache'+datetime.now().strftime("%H_%M_%S")+'.wav'
    audio, fs = torchaudio.load(uploaded_audio)
    audio += 1e-4 - torch.mean(audio)
    # torchaudio.save(cache_filename, audio, fs)
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Length", value=audio.size(-1))
    col2.metric(label="Sampling Rate", value=fs)
    col3.metric(label="File Type", value=splitext(uploaded_audio.name)[1])
    st.audio(uploaded_audio)

    spectrogram = torchaudio.functional.spectrogram(audio,0,torch.hann_window(512),512,256,512,2,False).squeeze()
    spectrogram = torchaudio.functional.amplitude_to_DB(spectrogram.abs(),20,1e-5,1).squeeze(0)
    sp_fig, sp_ax = plt.subplots()
    x_ticks = np.linspace(0, spectrogram.size(1),8)
    x_lable = np.round(x_ticks*256/fs,2)
    y_ticks = np.linspace(0,spectrogram.size(0)-1,9)
    y_lable = y_ticks*fs/256
    sp_ax.pcolormesh(spectrogram.numpy(), cmap='PuBu_r')
    plt.xticks(x_ticks,x_lable)
    plt.yticks(y_ticks,y_lable)
    plt.ylim([0,256])
    st.markdown('# Spectrogram')
    st.pyplot(sp_fig)

    @st.cache(allow_output_mutation=True)
    def load_model():
        model_path = './checkpoints/vctk_hifitts_G4A3L3_56ngf_6x_4_save2/model_scripted.pt'
        opt_path = './checkpoints/vctk_hifitts_G4A3L3_56ngf_6x_4_save2/opt.pt'
        opt = torch.load(opt_path)
        opt.phase = 'test'
        opt.snr = 55
        opt.gen_overlap = 0
        opt.is_lr_input = fs<=8000
        model = torch.jit.load(model_path)
        model.eval()
        return model, opt
    
    with st.spinner('Loading model'):
        model, opt = load_model()
    st.success('Load model done!')
    
    @st.cache(allow_output_mutation=True)
    def load_data():
        from data.audio_dataset import AudioAppDataset
        dataset = AudioAppDataset(opt, audio, fs)
        data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=48,
                num_workers=1,
                shuffle=False,
                pin_memory=True)
        return data_loader

    with st.spinner('Loading data'):
        data_loader = load_data()
    st.success('Load data done!')
    
    def inference():
        sr_audios = []
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                sr_spectro, sr_audio, _, _, _ = model.inference(
                    data['label'])
                print(sr_spectro.size())
                if opt.gen_overlap > 0:
                    sr_audio[...,:opt.gen_overlap] *= 0.5
                    sr_audio[...,-opt.gen_overlap:] *= 0.5
                sr_audios.append(sr_audio)
        # Concatenate the audio
        stride = opt.segment_length-opt.gen_overlap
        if opt.gen_overlap > 0:
            from torch.nn.functional import fold
            sr_audio = torch.cat(sr_audios,dim=0)
            out_len = (sr_audio.size(0)-1) * stride + opt.segment_length
            sr_audio = sr_audio.squeeze().transpose(-1,-2)
            print(sr_audio.shape)
            sr_audio = fold(sr_audio, kernel_size=(1,opt.segment_length), stride=(1,stride), output_size=(1,out_len)).squeeze(0)
            sr_audio = 2*sr_audio.cpu()[...,opt.gen_overlap//2:-opt.gen_overlap//2]
        else:
            sr_audio = torch.cat(sr_audios, dim=0).view(1, -1).cpu()
        results_file = BytesIO()
        torchaudio.save(results_file, sr_audio.float(), 48000, format='wav')
        return results_file, sr_audio
    with st.spinner('processing'):
        results_file, sr_audio = inference()
    st.success('Done!')

    st.title('Reconstructed Audio')
    st.audio(results_file)
    st.download_button(
     label="Download audio",
     data=results_file,
     file_name='reconstructed.wav',
     mime='audio/wav')
    
    sr_spectrogram = torchaudio.functional.spectrogram(sr_audio,0,torch.hann_window(512),512,256,512,2,False).squeeze()
    sr_spectrogram = torchaudio.functional.amplitude_to_DB(sr_spectrogram.abs(),20,1e-5,1).squeeze(0)
    sr_sp_fig, sr_sp_ax = plt.subplots()
    x_ticks = np.linspace(0, sr_spectrogram.size(1),8)
    x_lable = np.round(x_ticks*256/48000,2)
    y_ticks = np.linspace(0,sr_spectrogram.size(0)-1,9)
    y_lable = y_ticks*48000/256
    sr_sp_ax.pcolormesh(sr_spectrogram.numpy(), cmap='PuBu_r')
    plt.xticks(x_ticks,x_lable)
    plt.yticks(y_ticks,y_lable)
    plt.ylim([0,256])
    st.markdown('# Reconstructed Spectrogram')
    st.pyplot(sr_sp_fig)
