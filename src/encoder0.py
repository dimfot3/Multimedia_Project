import numpy as np
import subband_utils
from scipy.io import wavfile


def codec0(wavin, h, M, N):
    """
    codec0 is the core of MP3 implementation. It is a 
    simple format that only implements the subband filtering.
    It encodes and decodes the data.

    :param wavin: the path the sound file
    :param h: the prototype filter response
    :param M: the number of filters
    :param N: the length of each subband
    :return: the decoded sound and the encoded frames as KxNxM array where K 
    the number of frames
    """ 
    H = subband_utils.make_mp3_analysisfb(h, M)
    samplerate, data = wavfile.read(wavin)
    data_len = data.shape[0]
    data = np.append(data, np.zeros((M*N, )), axis=0)
    Y_tot = np.zeros((0, N, M))
    x_buff = np.zeros((M * (N-1) + H.shape[0]))
    for i in range(data_len // (M*N)):
        x_buff = data[0: M * (N-1) + H.shape[0]]
        y_frame = subband_utils.frame_sub_analysis(x_buff, H, N)
        Y_c = subband_utils.donothing(y_frame).reshape(-1, N, M)
        Y_tot = np.append(Y_tot, Y_c, axis=0)
        data = np.roll(data, - M * N)

    G = subband_utils.make_mp3_synthesisfb(h, M)
    x_hat = np.zeros((Y_tot.shape[0]*N*M, ))
    y_frame_buff = np.zeros((N + G.shape[0] // M, M))
    Y_arr = np.vstack([Y for Y in Y_tot])
    Y_arr = np.append(Y_arr, np.zeros((G.shape[0] // M, M)), axis=0)
    for fr_i in range(0, Y_tot.shape[0]):
        y_frame_buff = subband_utils.idonothing(Y_arr[0: N + G.shape[0] // M])
        x_hat[fr_i*M*N:(fr_i+1)*M*N] = subband_utils.frame_sub_synthesis(y_frame_buff, G)
        Y_arr = np.roll(Y_arr, -N, axis=0)
    return x_hat.reshape(-1, ), Y_tot


def coder0(wavin, h, M, N):
    """
    coder0 is implementing the encoding of a sound file. This is a
    first version that only implements the subband filtering.

    :param wavin: the path the sound file
    :param h: the prototype filter response
    :param M: the number of filters
    :param N: the length of each subband
    :return: encoded frames as KxNxM array where K 
    the number of frames
    """ 
    H = subband_utils.make_mp3_analysisfb(h, M)
    samplerate, data = wavfile.read(wavin)
    data_len = data.shape[0]
    data = np.append(data, np.zeros((M*N, )), axis=0)
    Y_tot = np.zeros((0, N, M))
    x_buff = np.zeros((M * (N-1) + H.shape[0]))
    for i in range(data_len // (M*N)):
        x_buff = data[0: M * (N-1) + H.shape[0]]
        y_frame = subband_utils.frame_sub_analysis(x_buff, H, N)
        Y_c = subband_utils.donothing(y_frame).reshape(-1, N, M)
        Y_tot = np.append(Y_tot, Y_c, axis=0)
        data = np.roll(data, - M * N)
    return Y_tot

def decoder0(Y_tot, h, M, N):
    """
    decoder0 is implementing the decoding of encoded sound file. This
    is a prototype version and implements only the inverse of subband filtering.
    :param wavin: the path the sound file
    :param h: the prototype filter response
    :param M: the number of filters
    :param N: the length of each subband
    :return: the decoded sound
    """
    G = subband_utils.make_mp3_synthesisfb(h, M)
    x_hat = np.zeros((Y_tot.shape[0]*N*M, ))
    y_frame_buff = np.zeros((N + G.shape[0] // M, M))
    Y_arr = np.vstack([Y for Y in Y_tot])
    Y_arr = np.append(Y_arr, np.zeros((G.shape[0] // M, M)), axis=0)
    for fr_i in range(0, Y_tot.shape[0]):
        y_frame_buff = subband_utils.idonothing(Y_arr[0: N + G.shape[0] // M])
        x_hat[fr_i*M*N:(fr_i+1)*M*N] = subband_utils.frame_sub_synthesis(y_frame_buff, G)
        Y_arr = np.roll(Y_arr, -N, axis=0)
    return x_hat

