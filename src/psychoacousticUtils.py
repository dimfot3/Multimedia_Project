import numpy as np
from time import time


def DCTpower(c):
    """
    DCTpower takes the dct coefficients of a frame and return their power in
    dB.

    :param c: DCT coefficients
    :return: power of dct coefs in dB.
    """
    return 20*np.log10(np.abs(c))

def Dksparse(Kmax):
    """
    Dksparse calculates a neighbors' matrix M for every frequency. 
    If one frequency j is at k's neibor then M(k, j) = 1 else 0.

    :param Kmax: the maximum frequency
    :return: the neighbors' matrix as Kmax x Kmax matrix
    """
    Dk = np.zeros((Kmax, Kmax), dtype=int)
    arr_k_idxs = np.repeat(np.array(np.arange(Kmax).reshape(-1, 1)), Kmax, axis=1)
    arr_j_idxs = np.repeat(np.array(np.arange(Kmax).reshape(1, -1)), Kmax, axis=0)
    idx_dist_arr = np.abs(arr_k_idxs - arr_j_idxs)
    Dk[(2 < arr_k_idxs) & (arr_k_idxs < 282) & (idx_dist_arr == 2)] = 1
    Dk[(282 <= arr_k_idxs) & (arr_k_idxs < 570) & (2 <= idx_dist_arr) & (idx_dist_arr <= 13)] = 1
    Dk[(570 <= arr_k_idxs) & (arr_k_idxs < 1152) & (2 <= idx_dist_arr) & (idx_dist_arr <= 27)] = 1
    return Dk

def STinit(c, D):
    """
    STinit finds the tones of a frame based on power of dct coefficients and 
    the matrix of neighbors.

    :param c: the DCT coefficients
    :param D: the neighbors matrix
    :return: the powerful tones
    """
    c_power = DCTpower(c)
    D = D.astype(bool)
    ST_arr = np.zeros((D.shape[0], )).astype(bool)
    for k in range(D.shape[0]):
        idx_t1 = np.array(np.clip([k - 1, k + 1], 0, D.shape[0] - 1))           # k + 1 and k - 1 idxs
        idx_t1 = idx_t1[idx_t1!=k]
        idx_t2 = np.argwhere(D[k, :]==1, ).reshape(-1, )    
        term1 = (c_power[k] > c_power[idx_t1]).all()
        term2 = (c_power[k] > c_power[idx_t2] + 7).all()
        ST_arr[k] = (term1 and term2)
    ST_arr = np.argwhere(ST_arr).reshape(-1, )
    return ST_arr

def MaskPower(c, ST):
    """
    MaskPower finds the power in dB of the mask tones

    :param c: the DCT coefficients
    :param ST: the mask tones indexes
    :return: the power of the mask tones
    """
    c = np.insert(c, [0, c.shape[0]], 0)        # this padding makes valid the (0 - 1) and (c.shape[0] - 1 + 1)
    ST = ST + 1         # this takes into account the zero padding
    power_tones = 10 * np.log10(c[ST]**2 + c[ST - 1]**2 + c[ST + 1]**2)
    return power_tones

def Hz2Barks(f):
    """
    Hz2Barks transforms the frequencies from Hz to Barks

    :param f: the frequencies in Hz
    :return: the frequencies in Barks
    """
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500)**2)

def STreduction(ST, c, Tq, fs=44100):
    """
    STreduction reduce the number of tones. It keeps only those above 
    the acoustic threshold at silence and from those that are
    closer than 0.5 barks keeps the one with highest power.

    :param ST: the masking tones
    :param c: the dct coefficients
    :param Tq: the Tq acoustic threshold at silence
    :param fs: the sample rate (default=44100)
    :return: the frequencies in Barks
    """
    pm_st = MaskPower(c, ST)
    ST_f = ST[pm_st >= Tq[ST]]
    pm_f = pm_st[pm_st >= Tq[ST]]
    barks = Hz2Barks(np.arange(c.shape[0]) * fs / (2 * c.shape[0]))
    barks_st = barks[ST_f]
    barks_diff = np.abs(barks_st.reshape(-1, 1) - barks_st.reshape(1, -1))
    ST_out = np.array([], dtype='int')
    pm_out = np.array([])
    for i in range(ST_f.shape[0]):
        barks_diff_st = barks_diff[i]
        close_st = barks_diff_st < 0.5
        if (pm_f[i] >= pm_f[close_st]).all():
            ST_out = np.append(ST_out, ST_f[i])
            pm_out = np.append(pm_out, pm_f[i])
    return ST_out, pm_out

def SpreadFunc(ST, PM, Kmax, fs=44100):
    """
    SpreadFunc calculate the spreading effect of a masking tone

    :param ST: the masking tones
    :param PM: the dct coefficients's power of masking tones
    :param Kmax: the maximum discrete frequency
    :param fs: the sample rate (default=44100)
    :return: an array (Kmax + 1) x len(ST) which contains for every column j the 
    effect of the masking tone j in each discrete frequency
    """
    spr_arr = np.zeros((Kmax + 1, ST.shape[0]))
    barks = Hz2Barks(np.arange(spr_arr.shape[0])*fs/(2*spr_arr.shape[0]))
    for i in range(Kmax + 1):
        for j in range(ST.shape[0]):
            dz = barks[i] - barks[ST[j]]
            if (dz >= -3) and (dz < -1):
                spr_arr[i, j] = 17 * dz - 0.4 * PM[j] + 11
            elif (dz >= -1) and (dz < 0):
                spr_arr[i, j] = (0.4 * PM[j] + 6) * dz
            elif (dz >= 0) and (dz < 1):
                spr_arr[i, j] = - 17 * dz
            elif (dz >= 1) and (dz < 8):
                spr_arr[i, j] = (0.15 * PM[j] - 17) * dz - 0.15 * PM[j]
    return spr_arr

def Masking_Thresholds(ST, PM, Kmax, fs=44100):
    """
    Masking_Thresholds calculates the effect of each masking tone
    in acoustic threshold

    :param ST: the masking tones
    :param PM: the dct coefficients
    :param Kmax: the maximum discrete frequency
    :param fs: the sample rate (default=44100)
    :return: an array (Kmax + 1) x len(ST) which contains for every column j the 
    the effect of j masking tone in acoustic threshold
    """
    spf = SpreadFunc(ST, PM, Kmax, fs=fs)
    tm = np.zeros((Kmax + 1, ST.shape[0]))
    barks = Hz2Barks(np.arange(Kmax + 1) * fs / (2 * (Kmax + 1)))
    for i in range(Kmax + 1):
        for j in range(ST.shape[0]):
            tm[i, j] = PM[j] - 0.275 * barks[j] + spf[i, j] - 6.025
    return tm

def Global_Masking_Thresholds(Ti, Tq):
    """
    Global_Masking_Thresholds calculates the total acoustic threshold 
    taking into acount the acoustic threshold effect of masking tones and
    acoustic threshold at silence.

    :param Ti: the masking tones
    :param Tq: the acoustic threshold at silence
    :return: An array of len(Tq) that contain the total acoustic threshold at dB
    """
    invlog_Tq = np.power(10, 0.1 * Tq)        # acoustic threshold at silence Power
    sum_invlog_Ti = np.sum(np.power(10, 0.1 * Ti), axis=1)        # total effect of masking tones
    Tg = 10 * np.log10(invlog_Tq + sum_invlog_Ti)     # acoustic threshold at silence + total effect of masking tones (dB)
    return Tg

def psycho(c, D, Tq_path='./protocol_files/Tq.npy'):
    """
    psycho implements the total psycoacoustic model

    :param c: the dct coefficients
    :param D: calculates a neighbors' matrix M for every frequency 
    :param Tq_path: the path to acoustic threshold at silence (default: './protocol_files/Tq.npy')
    :return: the total acoustic threshold
    """
    Tq = np.load(Tq_path)[0]
    Tq[np.isnan(Tq)] = Tq.max()
    ST = STinit(c, D)
    ST_red, ST_power = STreduction(ST, c, Tq)
    Tm = Masking_Thresholds(ST_red, ST_power, c.shape[0] - 1)
    Tg = Global_Masking_Thresholds(Tm, Tq)
    return Tg