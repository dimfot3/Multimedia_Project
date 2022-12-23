
import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# this is for an annoying librosa warning
import warnings
warnings.filterwarnings('ignore')


class Audio:
    """
    @brief: This is a class that can represent an uncompressed audio. Basic functionalities are provided
    to give access to audio data and its info.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data, self.info = self.load_sound(file_name)

    def load_sound(self, file_name):
        y, sr = librosa.load(file_name)
        duration = librosa.get_duration(y=y, sr=sr)
        info = {'sr':sr, 'dur':duration}
        return y, info

    def play(self):
        sd.play(self.data, self.info['sr'], blocking=True)

    def get_bitrate(self, bps=16):
        return self.info['sr'] * bps

    def get_size_bytes(self, bps=16):
        return self.get_bitrate(bps) * self.info['dur'] / 8

    def print_info(self, bps=16):
        print(f'Audio file: {self.file_path}\nnumber of samples: {self.data.shape[0]}' +
        f'\nbit rate(b/s): {self.get_bitrate(bps)}\naudio size(KB): {self.get_size_bytes(bps)/1024}'+
        f'\nbits per sample: {bps}')


file_name = './data/911-whats-your-emergency-104104.mp3'
audio = Audio(file_name)
audio.print_info(16)