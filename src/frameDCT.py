import numpy as np
from scipy.fftpack import dct, idct


def frameDCT(Y):
    Y = Y.T
    frame_coefs = np.zeros((Y.shape[0]*Y.shape[1]))
    for i, band in enumerate(Y):
        frame_coefs[i*Y.shape[1]:(i+1)*Y.shape[1]] = dct(band, norm='ortho')
    frame_coefs = frame_coefs.reshape(-1, )
    return frame_coefs

def iframeDCT(Y, M=32, N=36):
    frame = np.zeros((N, M))
    for i in range(M):
        frame[:, i] = idct(Y[i*N:(i+1)*N], norm='ortho')
    return frame