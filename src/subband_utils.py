import numpy as np

def make_mp3_analysisfb(h: np.ndarray, M: int) -> np.ndarray:
    """
    make_mp3_analysisfb produces the filter responses, used by
    analysis of subbands using the response of the prototype 
    fiter of length N.

    :param h: the prototype filter response
    :param M: the number of filters
    :return: returns the response of the filters in an numpy 
    array NxM.
    """
    H = np.zeros([len(h), M], dtype=np.float32)
    for i in range(1, M + 1):
        n = np.arange(h.shape[0], dtype=np.int64)
        freq_i = (2 * i - 1) * np.pi / (2.0 * M)
        phas_i = -(2 * i - 1) * np.pi / 4.0
        tmp = np.cos(freq_i * n + phas_i)
        x = np.multiply(h, tmp)
        H[:, i - 1] = x
    return H

def make_mp3_synthesisfb(h: np.ndarray, M: int) -> np.ndarray:
	"""
    make_mp3_synthesisfb produces the filter responses, used by
    synthesis of subbands using the response of the prototype 
    fiter of length N.

    :param h: the prototype filter response
    :param M: the number of filters
    :return: returns the response of the filters in an numpy 
    array NxM.
    """
	H = make_mp3_analysisfb(h, M)
	L = len(h)
	G = np.flip(H, axis=0)
	return G


def frame_sub_analysis(xbuff: np.ndarray, H: np.ndarray, q: int) -> np.ndarray:
    """
    frame_sub_analysis makes the subband analysis of a frame.

    :param xbuff: the buffer with length (q-1) x M + L, where q  the length of each subband,
    M the number of subbands and L the length of the filters's response. It contains in the left the 
    newest elements and right the the oldest.
    :param H: the filter's responses
    :param q: the length of each subband 
    :return: a frame with length NxM containing the subbands
    """
    L, M = H.shape
    ind = np.zeros([q, L])
    ind[0, :] = np.arange(L)

    for i in range(1, q):
        ind[i, :] += ind[i - 1, :] + M
    ind = ind.astype(np.int64)
    X = xbuff[ind]
    Y = np.einsum('ik,kj->ij', X, H)
    return Y


def frame_sub_synthesis(ybuff: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    frame_sub_analysis makes the subband synthesis of a frame.

    :param ybuff: this buffer contains M subbands of length (N-1 + L/M). The 
    format of the buffer is (N-1 + L/M) x M and the upper elements in each
    subband are the newest while the others are the oldest
    :param G: the filter's responses that reverse subband analysis
    :return: M * N samples where right contains the newest and left the oldest element
    """
    L, M = G.shape
    N = int(np.ceil(L / M))
    Gr = G.reshape(M, M * N, order='F').copy()
    Z = np.zeros([1152])
    for n in range(ybuff.shape[0] - N):
        tmp = ybuff[n:n + N, :].T.flatten()
        yr = np.expand_dims(tmp, axis=-1)
        z = np.dot(Gr, yr)
        Z[n * M:(n + 1) * M] = M * np.flip(z[:, 0])
    return Z.T.flatten()


def donothing(Yc: np.ndarray) -> np.ndarray:
    """
    donothing this is a dummy function that simulates the frame
    processing.

    :param Yc: the frame after subband analysis
    :return: the frame after the processing
    """
    return Yc

def idonothing(Yc: np.ndarray) -> np.ndarray:
    """
    idonothing this is a dummy function that simulates the
    inverse of frame processing.

    :param Yc: the processed frame
    :return: the frame after inversing the processing, in subbands format
    """
    return Yc


