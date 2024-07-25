import math

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class SoundDataset(Dataset):

    def __init__(self, noisy_files, clean_files, n_fft=64, hop_length=16):
        super().__init__()

        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.len_ = len(self.noisy_files)

        self.max_len = 165000


    def __len__(self):
        return self.len_

    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform

    def __getitem__(self, index):
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])

        x_clean_chunks = self._prepare_sample(x_clean)
        x_noisy_chunks = self._prepare_sample(x_noisy)


        for i in range(len(x_clean_chunks)):
            x_clean_chunks[i] = torch.stft(input=x_clean_chunks[i], n_fft=self.n_fft,
                                  hop_length=self.hop_length, normalized=True, return_complex=True, window=torch.hann_window(window_length=3072).to(x_clean.device))
        for i in range(len(x_noisy_chunks)):
            x_noisy_chunks[i] = torch.stft(input=x_noisy_chunks[i], n_fft=self.n_fft,
                                  hop_length=self.hop_length, normalized=True, return_complex=True, window=torch.hann_window(window_length=3072).to(x_noisy.device))

        return x_noisy_chunks, x_clean_chunks

    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        num_chunks = math.ceil(waveform.shape[1] / self.max_len)
        chunks = []

        for i in range(num_chunks):
            start = i * self.max_len
            end = min((i + 1) * self.max_len, waveform.shape[1])
            chunk = waveform[:, start:end]
            if chunk.shape[1] < self.max_len:
                # Padding if necessary
                chunk = np.pad(chunk, ((0, 0), (0, self.max_len - chunk.shape[1])), mode='constant')

            chunks.append(torch.from_numpy(chunk))

        return chunks
