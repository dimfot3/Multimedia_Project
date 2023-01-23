import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src/')
from frameDCT import frameDCT
from encoder0 import coder0
from psychoacousticUtils import *


music_file = './data/myfile.wav'
h_pr = np.load('./protocol_files/h.npy', allow_pickle=True).tolist()['h'].reshape(-1, )
Tq = np.load('./protocol_files/Tq.npy')[0]
Tq[np.isnan(Tq)] = Tq[~np.isnan(Tq)].max()
M = 32
N = 36
fs = 44100
Y_tot = coder0(music_file, h_pr, M, N)


frames_to_plot = 10
rand_frames = np.random.randint(0, Y_tot.shape[0], frames_to_plot)
for i, frame in enumerate(Y_tot[rand_frames]):
    frame_coef = frameDCT(frame)
    Dk_mat = Dksparse(M*N)
    Tg = psycho(frame_coef, Dk_mat)
    plt.plot(np.arange(frame_coef.shape[0])*fs/(2*M*N), Tg)
    plt.title(f'Total acoustic threshold at frame:{rand_frames[i]}', fontsize=20, fontweight='bold')
    plt.xlabel('Hz', fontsize=20)
    plt.ylabel('Power(dB)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
