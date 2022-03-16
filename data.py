import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import glob
import random
import os




class EmovDB(Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode
        if self.mode == 'all':
            self.mels = glob.glob(f'./data/spectrograms/**/*.npy', recursive=True)
        else:
            self.mels = glob.glob(f'./spectrograms/{mode}/**/*.npy')
        self.labels = {'l': 0, 'e': 1, 'i': 2, 'n': 3, 'm': 4}

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        mel_path = self.mels[idx]
        mel = np.load(mel_path)
        mel = mel[np.newaxis, ...]

        label = 0 # self.labels[mel_path.split('/')[-2][1]] # label is needed only for classification
        # if self.transform:
        #     return None
        return mel, label

    def transform(self, audio):
        """
        Performs the transformation of the mel filter and the STFT
        representation of the audio data
        """
        return audio

    def standard_normal_variate(self, data):
        mean = np.mean(data)
        std = np.std(data)

        return (data - mean) / std

    def min_max_scaler(self, data):
        # print('data max', np.max(data))
        # print('data min', np.min(data))
        return (data - np.min(data))/((np.max(data) - np.min(data))+1e-10)