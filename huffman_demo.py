import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src/')
from frameDCT import frameDCT
from encoder0 import coder0
from psychoacousticUtils import *
from quantizer_utils import *
from time import time
from tqdm import tqdm
from rle_utils import *
from huffman_utils import huff, ihuff

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
for i, frame in enumerate(Y_tot[rand_frames]):
    frame_coef = frameDCT(frame)
    Tg = psycho(frame_coef, Dk_mat)
    symb_index, SF, B = all_bands_quantizer(frame_coef, Tg)
    rle_out = RLE(symb_index, M*N)
    bts, table = huff(rle_out)
    new_rle_out = ihuff(bts, table)
    print('frame', rand_frames[i],'Huffman inversed:', (rle_out == new_rle_out).all())
    symb_index_new = RLE_inv(new_rle_out, M*N)
    print('frame', rand_frames[i], 'RLE inversed:', (symb_index == symb_index_new).all())


