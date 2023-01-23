import numpy as np
import sys 
sys.path.insert(0, './src/')
from src.MP3_total_pipeline import MP3codec, MP3_cod, MP3_decod
from scipy.io import wavfile
import matplotlib.pyplot as plt

# load the prototype response
h_pr = np.load('./protocol_files/h.npy', allow_pickle=True).tolist()['h'].reshape(-1, )
M = 32
N = 36
music_file = './data/wav_2mb.wav'
samplerate, data = wavfile.read(music_file)

#xhat = MP3codec(data, h_pr, M, N)
MP3_cod(data, h_pr, M, N)
xhat = MP3_decod('./outputs/bitstream.bin', './outputs/add_info.npy', h_pr, M, N)
wavfile.write('./outputs/decoded.wav', samplerate, xhat.astype('int16'))

