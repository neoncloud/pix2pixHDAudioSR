import streamlit as st
import torch
import torchaudio
import matplotlib.pyplot as plt
from os.path import splitext
from io import BytesIO

st.title('Audio Upload')
uploaded_audio = st.file_uploader("Please upload a audio file", ['wav', 'mp3', 'flac', 'ogg'], False)
if uploaded_audio is not None:
    #metadata = torchaudio.info(uploaded_audio)
    #st.write('Audio Length:',metadata.num_frames)
    #cache_filename = './cache'+datetime.now().strftime("%H_%M_%S")+'.wav'
    audio, fs = torchaudio.load(uploaded_audio)
    # torchaudio.save(cache_filename, audio, fs)
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Length", value=audio.size(-1))
    col2.metric(label="Sampling Rate", value=fs)
    col3.metric(label="File Type", value=splitext(uploaded_audio.name)[1])
    st.audio(uploaded_audio)

    spectrogram = torchaudio.functional.spectrogram(audio,0,torch.hann_window(512),512,256,512,2,False).squeeze()
    spectrogram = torchaudio.functional.amplitude_to_DB(spectrogram.abs(),20,1e-5,1).squeeze(0)
    sp_fig, sp_ax = plt.subplots()
    sp_ax.pcolormesh(spectrogram.numpy(), cmap='PuBu_r')
    st.markdown('# Spectrogram')
    st.pyplot(sp_fig)

    @st.cache(allow_output_mutation=True)
    def load_model():
        model_path = './checkpoints/vctk_hifitts_G4A3L3_56ngf_6x_4_save/model_scripted.pt'
        opt_path = './checkpoints/vctk_hifitts_G4A3L3_56ngf_6x_4_save/opt.pt'
        opt = torch.load(opt_path)
        opt.phase = 'test'
        opt.snr = 72
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
                batch_size=20,
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
                if opt.gen_overlap == 0:
                    sr_audios.append(sr_audio)
                else:
                    sr_audios.append(sr_audio[..., opt.gen_overlap//2:-opt.gen_overlap//2])
        sr_audio = torch.cat(sr_audios, dim=0).view(1, -1).to(torch.float32).cpu()
        results_file = BytesIO()
        torchaudio.save(results_file, sr_audio, fs, format='wav')
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
    sr_sp_ax.pcolormesh(sr_spectrogram.numpy(), cmap='PuBu_r')
    st.markdown('# Reconstructed Spectrogram')
    st.pyplot(sr_sp_fig)
