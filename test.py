import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import sys
sys.path.insert(0, 'src')
from Audio import Audio


class MP3Encoder:
    def __init__(self, protocol_files_path='./protocol_files/'):
        coefs = np.loadtxt(f'{protocol_files_path}C_coefs.txt')
        self.n_subband = 32
        self.frame_size = 1152
        self.subbants_len = 1024
        return

    def encode(self, audio):
        data = audio.data
        split_idxs = [min(self.subbants_len * i, data.shape[0]) for i in range(1, np.ceil(data.shape[0]/self.subbants_len).astype(int))]
        frame_arr = np.split(data, split_idxs, axis=0)
        self.filterbank(frame_arr[40])

    def filterbank(self, data):
        sample_splitted = np.array_split(data, self.n_subband)
        for i, data_part in enumerate(sample_splitted):
            x = np.hstack(np.repeat(data_part[np.newaxis, :], 16, axis=0))
            


file_name = './data/911-whats-your-emergency-104104.mp3'
audio = Audio(file_name)
#audio.print_info(16)

encoder = MP3Encoder()
encoder.encode(audio)


