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


frames_to_plot = 2
rand_frames = np.random.randint(0, Y_tot.shape[0], frames_to_plot)
for i, frame in enumerate(Y_tot[rand_frames]):
    frame_coef = frameDCT(frame)
    # power_coef = DCTpower(frame_coef)
    # Dk_mat = Dksparse(M*N)
    # ST = STinit(frame_coef, Dk_mat)
    # powerST = MaskPower(frame_coef, ST)
    # z_barks = Hz2Barks(np.arange(frame_coef.shape[0])*fs/(2*frame_coef.shape[0]))
    # ST_red, ST_power = STreduction(ST, frame_coef, Tq)
    # sp_f = SpreadFunc(ST_red, ST_power, frame_coef.shape[0] - 1)
    # tm = Masking_Thresholds(ST_red, ST_power, frame_coef.shape[0] - 1)
    # Tg = Global_Masking_Thresholds(tm, Tq)
    Dk_mat = Dksparse(M*N)
    Tg = psycho(frame_coef, Dk_mat)
    plt.plot(Tg)
    plt.show()
    # plt.plot(z_barks, power_coef)
    # if ST.shape[0]>0:
    #     plt.stem(z_barks[ST], power_coef[ST], linefmt ='red', label='Tone')
    # plt.xlabel('Barks', fontsize=20)
    # plt.ylabel('Power(dB)', fontsize=20)
    # plt.title(f'Power spectrume frame:{rand_frames[i]}', fontsize=20, fontweight='bold')
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.legend()
    # plt.show()
    
