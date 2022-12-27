import numpy as np
import utils
from scipy.io import wavfile

def codec0(wavin, h, M, N):
    H = utils.make_mp3_analysisfb(h, M)
    samplerate, data = wavfile.read(wavin)
    split_idxs_arr = utils.split_arrays_equally(np.arange(data.shape[0]), M*N)
    Y_tot = np.zeros((len(split_idxs_arr), N, M))
    for i, idxs in enumerate(split_idxs_arr):
        x_frame = data[idxs]
        x_frame_buf = np.repeat(x_frame[np.newaxis, :], N*(M-1) + H.shape[0], axis=0).reshape(-1, )
        y_frame = utils.frame_sub_analysis(x_frame_buf, H, N)
        Y_c = utils.donothing(y_frame)
        Y_tot[i] = Y_c

    G = utils.make_mp3_synthesisfb(h, M)
    x_hat = np.zeros((Y_tot.shape[0], N*M))
    for fr_i in range(Y_tot.shape[0]):
        y_frame = Y_tot[fr_i]
        y_dc = utils.idonothing(y_frame)
        k = int((N - 1) + G.shape[0]/M)
        y_frame_buff = np.zeros((k, M))
        for filt_idx in range(M):
            y_frame_buff[:, filt_idx] = np.repeat(y_dc[:, filt_idx][np.newaxis, :], int(np.ceil(k/N)), axis=0).reshape(-1, )[:k]
        x_hat[fr_i] = utils.frame_sub_synthesis(y_frame_buff, G)
    return x_hat.reshape(-1, ), Y_tot


def coder0(wavin, h, M, N):
    H = utils.make_mp3_analysisfb(h, M)
    samplerate, data = wavfile.read(wavin)
    split_idxs_arr = utils.split_arrays_equally(np.arange(data.shape[0]), M*N)
    Y_tot = np.zeros((len(split_idxs_arr), N, M))
    for i, idxs in enumerate(split_idxs_arr):
        x_frame = data[idxs]
        x_frame_buf = np.repeat(x_frame[np.newaxis, :], N*(M-1) + H.shape[0], axis=0).reshape(-1, )
        y_frame = utils.frame_sub_analysis(x_frame_buf, H, N)
        Y_c = utils.donothing(y_frame)
        Y_tot[i] = Y_c
    return Y_tot

def decoder0(Y_tot, h, M, N):
    G = utils.make_mp3_synthesisfb(h, M)
    x_hat = np.zeros((Y_tot.shape[0], N*M))
    for fr_i in range(Y_tot.shape[0]):
        y_frame = Y_tot[fr_i]
        y_dc = utils.idonothing(y_frame)
        k = int((N - 1) + G.shape[0]/M)
        y_frame_buff = np.zeros((k, M))
        for filt_idx in range(M):
            y_frame_buff[:, filt_idx] = np.repeat(y_dc[:, filt_idx][np.newaxis, :], int(np.ceil(k/N)), axis=0).reshape(-1, )[:k]
        x_hat[fr_i] = utils.frame_sub_synthesis(y_frame_buff, G)
    return x_hat.reshape(-1, )

