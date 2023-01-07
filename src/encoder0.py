import numpy as np
import utils
from scipy.io import wavfile

def codec0(wavin, h, M, N):
    H = utils.make_mp3_analysisfb(h, M)
    samplerate, audiodata = wavfile.read(wavin)
    data = np.zeros((audiodata.shape[0] + audiodata.shape[0] % (M*N), ))
    data[:audiodata.shape[0]] = audiodata
    
    split_idxs_arr = utils.split_arrays_equally(np.arange(data.shape[0]), M*N)
    Y_tot = np.zeros((len(split_idxs_arr), N, M))
    x_frame_old = np.zeros(H.shape[0] - M, )
    for i, idxs in enumerate(split_idxs_arr):
        x_frame = data[idxs]
        x_frame_buf = np.append(np.flip(x_frame), x_frame_old, axis=0)
        x_frame_old = x_frame_buf[0:H.shape[0] - M]
        y_frame = utils.frame_sub_analysis(x_frame_buf, H, N)
        Y_c = utils.donothing(y_frame)
        Y_tot[i] = Y_c

    G = utils.make_mp3_synthesisfb(h, M)
    x_hat = np.zeros((Y_tot.shape[0], N*M))
    y_frame_old = np.zeros((G.shape[0] // M , M))
    for fr_i in range(Y_tot.shape[0]):
        y_frame = Y_tot[fr_i]
        y_dc = utils.idonothing(y_frame)
        y_frame_buff = np.append(y_frame_old, np.flip(y_dc, axis=0), axis=0)
        y_frame_old = y_frame_buff[y_frame_buff.shape[0] - y_frame_old.shape[0]:]
        x_hat[fr_i] = utils.frame_sub_synthesis(y_frame_buff, G)
    return x_hat.reshape(-1, ), Y_tot


def coder0(wavin, h, M, N):
    H = utils.make_mp3_analysisfb(h, M)
    samplerate, data = wavfile.read(wavin)
    split_idxs_arr = utils.split_arrays_equally(np.arange(data.shape[0]), M*N)
    Y_tot = np.zeros((len(split_idxs_arr), N, M))
    x_frame_old = np.zeros(H.shape[0] - M, )
    for i, idxs in enumerate(split_idxs_arr):
        x_frame = data[idxs]
        x_frame_buf = np.append(np.flip(x_frame), x_frame_old, axis=0)
        x_frame_old = x_frame_buf[0:H.shape[0] - M]
        y_frame = utils.frame_sub_analysis(x_frame_buf, H, N)
        Y_c = utils.donothing(y_frame)
        Y_tot[i] = Y_c
    return Y_tot

def decoder0(Y_tot, h, M, N):
    G = utils.make_mp3_synthesisfb(h, M)
    x_hat = np.zeros((Y_tot.shape[0], N*M))
    y_frame_old = np.zeros((G.shape[0] // M , M))
    for fr_i in range(Y_tot.shape[0]):
        y_frame = Y_tot[fr_i]
        y_dc = utils.idonothing(y_frame)
        y_frame_buff = np.append(y_frame_old, np.flip(y_dc, axis=0), axis=0)
        y_frame_old = y_frame_buff[y_frame_buff.shape[0] - y_frame_old.shape[0]:]
        x_hat[fr_i] = utils.frame_sub_synthesis(y_frame_buff, G)
    return x_hat.reshape(-1, )

