import numpy as np


def make_mp3_analysisfb(h: np.ndarray, M: int) -> np.ndarray:
    """
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
	"""
	H = make_mp3_analysisfb(h, M)
	L = len(h)
	G = np.flip(H, axis=0)
	return G


def frame_sub_analysis(xbuff: np.ndarray, H: np.ndarray, q: int) -> np.ndarray:
	"""
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
	return Yc


def idonothing(Yc: np.ndarray) -> np.ndarray:
	return Yc

def split_arrays_equally(array, eq_len):
    split_idx_arr = [array[i*eq_len:min((i+1)*eq_len, array.shape[0])] for i in range(np.ceil(array.shape[0]/eq_len).astype('int'))]
    return split_idx_arr


class MP3Encoder:
    def __init__(self, protocol_files_path='./protocol_files/'):
        coefs = np.loadtxt(f'{protocol_files_path}C_coefs.txt')
        self.n_subband = 32
        self.frame_size = 1152
        self.subbants_len = 1024
        return

    def encode(self, audio):
        data = audio.data
        split_idxs = [min(self.subbants_len * i, data.shape[0]) for i in range(1, np.ceil(data.shape[0]/self.subbants_len).astype(int))]
        frame_arr = np.split(data, split_idxs, axis=0)
        self.filterbank(frame_arr[40])

    def filterbank(self, data):
        sample_splitted = np.array_split(data, self.n_subband)
        for i, data_part in enumerate(sample_splitted):
            x = np.hstack(np.repeat(data_part[np.newaxis, :], 16, axis=0))