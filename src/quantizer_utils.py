import numpy as np


def critical_bands(K, fs=44100):
    cb = np.zeros((K, )).astype('int')
    fz = np.arange(K)*fs/(2*K)
    hz_bar = np.array([0, 100, 100, 200, 200, 300, 300,
    400, 400, 510, 510, 630, 630, 770, 
    770, 920, 920, 1080, 1080, 1270, 1270, 1480, 1480, 1720, 1720, 2000, 
    2000, 2320, 2320, 2700, 2700, 3150, 3150, 3700, 3700, 4400, 
    4400, 5300, 5300, 6400, 6400, 7700, 
    7700, 9500, 9500, 12000, 12000, 15500, 15500, 44100]).reshape(-1, 2)
    for i in range(hz_bar.shape[0]):
        cb[np.logical_and(fz >= hz_bar[i, 0], fz < hz_bar[i, 1])] = i
    return cb

def DCT_band_scale(c, fs=44100):
    cb = critical_bands(c.shape[0], fs=fs)
    cs, sc = np.zeros((c.shape[0], )), np.zeros((np.unique(cb).shape[0], ))
    for i, cb_i in enumerate(np.unique(cb)):
        cb_idxs = (cb == cb_i)
        band_c = c[cb_idxs]
        band_c34 = np.power(np.abs(band_c), 3/4)
        sc[i] = np.max(band_c34)
        cs[cb_idxs] = np.sign(band_c) * band_c34 / sc[i]
    return cs, sc

def quantizer(x, b):
    wb = 1 / (2**b - 1)
    bins_left = np.insert(np.linspace(-(2**(b - 1) - 1)*wb, -wb, 2**(b - 1) - 1), 0, -1)
    bins_right = np.append(np.linspace(wb, (2**(b - 1) - 1)*wb, 2**(b - 1) - 1), 1)
    bins = np.append(bins_left, bins_right)
    x_q = np.digitize(x, bins, right=False) - 1
    x_q[x_q==(2**b - 1)] = 2**b - 2
    x_q -= (2**(b - 1) - 1)
    return x_q.astype('int')

def dequantizer(symb_index, b):
    symb_index = symb_index - (2**(b - 1) - 1)
    wb = 1 / (2**b - 1)
    bins_left = np.insert(np.linspace(-(2**(b - 1) - 1)*wb, -wb, 2**(b - 1) - 1), 0, -1)
    bins_right = np.append(np.linspace(wb, (2**(b - 1) - 1)*wb, 2**(b - 1) - 1), 1)
    bins = np.append(bins_left, bins_right)
    bins_centers = (bins[:-1] + bins[1:]) / 2
    xh = bins_centers[symb_index.astype('int')]
    return xh

def all_bands_quantizer(c, Tg):
    cb = critical_bands(c.shape[0])
    cs, sc = DCT_band_scale(c)
    b = 1
    for b in range(1, 10):
        symb_index = quantizer(cs, b)
        xh = dequantizer(symb_index, b)
        cs_estimated = np.zeros((cs.shape[0], ))
        for bandi in range(sc.shape[0]):
            bandi_idxs = (cb == bandi)
            cs_estimated[bandi_idxs] = np.sign(xh[bandi_idxs]) * np.power(np.abs(xh[bandi_idxs] * sc[bandi]), 3/4)
        error_power = 20*np.log10(np.abs(cs_estimated - cs))
        if((error_power <= Tg).all()):
            break
    return symb_index, sc, b

def all_bands_dequantizer(symb_index, B, SF):
    cb = critical_bands(symb_index.shape[0])
    xh = dequantizer(symb_index, B)
    cs_estimated = np.zeros((symb_index.shape[0], ))
    for bandi in range(SF.shape[0]):
        bandi_idxs = (cb == bandi)
        cs_estimated[bandi_idxs] = np.sign(xh[bandi_idxs]) * np.power(np.abs(xh[bandi_idxs] * SF[bandi]), 3/4)
    return cs_estimated
    