import numpy as np
import sys 
sys.path.insert(0, 'src/')
from src.encoder1 import MP3cod

# load the prototype response
h_pr = np.load('./protocol_files/h.npy', allow_pickle=True).tolist()['h'].reshape(-1, )
M = 32
N = 36
music_file = './data/myfile.wav'

MP3cod(music_file, h_pr, M, N)


