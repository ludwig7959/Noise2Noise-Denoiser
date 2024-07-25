import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from dotenv import load_dotenv

from models import DCUnet20

load_dotenv()

SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', 48000))
N_FFT = int(os.getenv('N_FFT', 3072))
HOP_LENGTH = int(os.getenv('HOP_LENGTH', 768))

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


def preprocess_audio(file):
    waveform, sr = torchaudio.load(file)

    waveform = waveform.to(DEVICE)

    waveform_np = waveform.cpu().numpy()
    num_samples = waveform_np.shape[1]

    max_length = 165000
    stfts = []

    for start in range(0, num_samples, max_length):
        end = min(start + max_length, num_samples)
        segment = waveform[:, start:end]

        output = torch.zeros((1, max_length), dtype=torch.float32, device=DEVICE)
        output[0, :segment.shape[1]] = segment

        stft = torch.stft(input=output, n_fft=N_FFT,
                          hop_length=HOP_LENGTH, normalized=True,
                          return_complex=True, window=torch.hann_window(3072).to(DEVICE))
        stfts.append(stft)

    return stfts, sr



model_weights_path = "Weights/dc20_model_30.pth"

dcunet20 = DCUnet20(N_FFT, HOP_LENGTH).to(DEVICE)
optimizer = torch.optim.Adam(dcunet20.parameters())

weights = torch.load(model_weights_path, map_location=torch.device(DEVICE))

dcunet20.load_state_dict(weights)

input_path = input('노이즈를 제거할 오디오 파일들의 경로를 입력하세요: ')
output_path = input('출력물 경로를 입력하세요: ')

for file in os.listdir(input_path):
    if not file.endswith(".wav"):
        continue

    stfts, sr = preprocess_audio(os.path.join(input_path, file))
    waveforms = []
    for stft in stfts:
        stft = torch.unsqueeze(stft, 0)
        waveform = dcunet20.forward(stft)
        waveform_numpy = waveform.detach().cpu().numpy()

        if waveform_numpy.ndim == 2:
            waveform_numpy = waveform_numpy[0]  # 첫 번째 채널만 선택

        waveforms.append(waveform_numpy)
    final = np.concatenate(waveforms, axis=-1)
    sf.write(os.path.join(output_path, file), final, samplerate=sr)
