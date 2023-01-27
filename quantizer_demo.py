import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src/')
from frameDCT import frameDCT, iframeDCT
from encoder0 import coder0
from psychoacousticUtils import *
from quantizer_utils import *
from time import time
from tqdm import tqdm
from encoder0 import decoder0
from scipy.io import wavfile


music_file = './data/myfile.wav'
h_pr = np.load('./protocol_files/h.npy', allow_pickle=True).tolist()['h'].reshape(-1, )
Tq = np.load('./protocol_files/Tq.npy')[0]
Tq[np.isnan(Tq)] = Tq[~np.isnan(Tq)].max()
M = 32
N = 36
fs = 44100
Y_tot = coder0(music_file, h_pr, M, N)  # splitting to bands

frames_to_plot = 10
rand_frames = np.random.randint(0, Y_tot.shape[0], frames_to_plot)
Dk_mat = Dksparse(M*N)
Y_new = np.zeros(Y_tot.shape)
for i, frame in tqdm(enumerate(Y_tot), total=Y_tot.shape[0]):
    frame_coef = frameDCT(frame)
    Tg = psycho(frame_coef, Dk_mat) - 15
    symb_index, SF, B = all_bands_quantizer(frame_coef, Tg)
    xh = all_bands_dequantizer(symb_index, B, SF)
    Y_new[i] = iframeDCT(xh)

xhat = decoder0(Y_new, h_pr, M, N)
wavfile.write('final.wav', fs, xhat.astype('int16'))
