import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src/')
from frameDCT import frameDCT, iframeDCT
from encoder0 import coder0


music_file = './data/myfile.wav'
h_pr = np.load('./protocol_files/h.npy', allow_pickle=True).tolist()['h'].reshape(-1, )
M = 32
N = 36
fs = 44100
Y_tot = coder0(music_file, h_pr, M, N)


frames_to_plot = 2
for i, frame in enumerate(Y_tot[:frames_to_plot]):
    frame_coef = frameDCT(frame)
    plt.plot(np.arange(frame_coef.shape[0])*fs/(2*M*N), frame_coef)
    plt.xlabel('Hz', fontsize=20)
    plt.ylabel('DCT coef', fontsize=20)
    plt.title(f'DCT coefs frame {i}', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    Y_new = iframeDCT(frame_coef, M=32, N=36)
    print('MSE of dct and inverse dct:', np.mean((frame.reshape(-1, ) - Y_new.reshape(-1, ))**2))

