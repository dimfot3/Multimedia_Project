import numpy as np


def critical_bands(K, fs=44100):
    """
    critical_bands finds in which critical band each of K frequencies

    :param K: the number of frequencies
    :param fs: the sampling rate (Hz) (default: 44100)
    :return: an array that shows in which from 25 critical bands each frequency belongs.
    array NxM.
    """
    fz = np.arange(K) * fs / (2 * K)
    hz_bar = np.array([0, 100, 200, 300,
    400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 
    2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 
    7700, 9500, 12000, 15500, 44100])
    cb = np.digitize(fz, hz_bar) - 1
    return cb

def DCT_band_scale(c, fs=44100):
    """
    DCT_band_scale normalize the dct coefficients and return them alongside with
    the scale factors.

    :param c: the dct coefficients
    :param fs: the sampling rate (Hz) (default: 44100)
    :return: the normalize coeficients and their scale factor
    """
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
    """
    quantizer quantize the normalized coefficients of critical band
    with symmetric quantizer in 2^b - 1 zones

    :param x: the normalized dct coefficients
    :param b: the number of bits to be used for quantization
    :return: the dct coefficients quantized
    """
    wb = 1 / (2**b - 1)
    bins_left = np.insert(np.linspace(-(2**(b - 1) - 1)*wb, -wb, 2**(b - 1) - 1), 0, -1)
    bins_right = np.linspace(wb, (2**(b - 1) - 1)*wb, 2**(b - 1) - 1)
    bins = np.append(bins_left, bins_right)
    x_q = np.digitize(x, bins, right=False) - 1     # -1 makes the first symbol start from zero
    x_q -= (2**(b - 1) - 1)
    return x_q.astype('int')

def dequantizer(symb_index, b):
    """
    dequantizer dequantize the quantized normalized dct coefficients

    :param symb_index: the quantized symbols
    :param b: the number of bits to be used for quantization
    :return: the normalized dct coefficients unquantized
    """
    symb_index = np.copy(symb_index) + (2**(b - 1) - 1)      # make the symbols zeros indexed
    wb = 1 / (2**b - 1)
    bins_left = np.insert(np.linspace(-(2**(b - 1) - 1)*wb, -wb, 2**(b - 1) - 1), 0, -1)
    bins_right = np.append(np.linspace(wb, (2**(b - 1) - 1)*wb, 2**(b - 1) - 1), 1)
    bins = np.append(bins_left, bins_right)
    bins_centers = (bins[:-1] + bins[1:]) / 2
    xh = bins_centers[symb_index.astype('int')]
    return xh

def all_bands_quantizer(c, Tg):
    """
    all_bands_quantizer finds the appropriate quantizer for each
    critical subband and quantize it.

    :param c: the dct coefficients
    :param Tg: the total acoustic threshold
    :return: the quantized symbols, the scale factors for each critical band and the bits 
    used for each critical band.
    """
    max_b = 16
    cb = critical_bands(c.shape[0])
    cs, sc = DCT_band_scale(c)
    Bits_per_critband = np.zeros((sc.shape[0], ), dtype='int')
    symb_index = np.zeros((c.shape[0], ), dtype='int')
    for bandi in range(sc.shape[0]):
        bandi_idxs = (cb == bandi)
        cs_band = cs[bandi_idxs]
        sc_band = sc[bandi]
        for b in range(1, max_b):
            symb_index[bandi_idxs] = quantizer(cs_band, b)
            csh_band = dequantizer(symb_index[bandi_idxs], b)            
            c_estimated = np.sign(csh_band) * np.power(np.abs(csh_band) * sc_band, 4/3)
            error_power = 20 * np.log10(np.abs(c[bandi_idxs] - c_estimated))
            if((error_power <= Tg[bandi_idxs]).all()) or (b == max_b - 1):
                Bits_per_critband[bandi] = b
                break
    return symb_index, sc, Bits_per_critband

def all_bands_dequantizer(symb_index, B, SF):
    """
    all_bands_quantizer dequantized each critical band using 
    the correct quantizer and rescale the dct coefficients

    :param symb_index: the quantized symbols
    :param B: the bits used for each critical band
    :param SF: the scale factor for each critical band
    :return: the dct coefficients
    """
    cb = critical_bands(symb_index.shape[0])
    c_estimated = np.zeros((symb_index.shape[0], ))
    for bandi in range(SF.shape[0]):
        bandi_idxs = (cb == bandi)
        cs_h = dequantizer(symb_index[bandi_idxs], B[bandi])
        c_estimated[bandi_idxs] = np.sign(cs_h) * np.power(np.abs(cs_h) * SF[bandi], 4/3)
    return c_estimated
    