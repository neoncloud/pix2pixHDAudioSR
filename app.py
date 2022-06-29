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
    audio, fs = torchaudio.load(uploaded_audio)
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
        model_dict = torch.load('/root/TRUMelnet_TRUnet/exp/PW_NBDF_models/multiscale/final_model.pth',map_location=lambda storage, loc: storage.cuda())
        
        model = TRUnet(n_fft=1024, win_length=512, hop_length=128, sample_rate=16000)
        model.load_state_dict(model_dict)
        #model.to(device)
        model.eval()
        return model
    
    model = load_model()

    @st.cache(allow_output_mutation=True)
    def inference():
        # inference 
        #device = torch.device("cuda") 
        estimate_source = model(audio) ##breakpoint
        estimate_source = estimate_source.squeeze(1)
        #print(estimate_source.size())
        return estimate_source.cpu()

    results = inference()
    results_file = BytesIO()
    torchaudio.save(results_file, results, 16000, format="wav")
    st.title('Reconstructed Audio')
    st.audio(results_file)
    st.download_button(
     label="Download audio",
     data=results_file,
     file_name='reconstructed.wav',
     mime='audio/wav',
 )
