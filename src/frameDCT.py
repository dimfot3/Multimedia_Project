import numpy as np
from scipy.fftpack import dct, idct


def frameDCT(Y):
    """
    frameDCT takes the Y frame of length NxM and applies DCT to every
    one of M bands.

    :param Y: the path the sound file
    :return: the DCT coefficients in one vector of M*N
    """ 
    Y = Y.T
    frame_coefs = np.zeros((Y.shape[0]*Y.shape[1]))
    for i, band in enumerate(Y):
        frame_coefs[i*Y.shape[1]:(i+1)*Y.shape[1]] = dct(band, type=4, norm='ortho')
    frame_coefs = frame_coefs.reshape(-1, )
    return frame_coefs

def iframeDCT(Y, M=32, N=36):
    """
    iframeDCT reverse the DCT applied to each band

    :param Y: this is the DCT coefficients of each band
    :param M: the number of bands (default: 32)
    :param N: the number of elements in each band (default: 36)
    :return: The Y frame in time as MxN array
    """ 
    frame = np.zeros((N, M))
    for i in range(M):
        frame[:, i] = idct(Y[i*N:(i+1)*N], type=4, norm='ortho')
    return frame