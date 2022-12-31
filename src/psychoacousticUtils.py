import numpy as np
from time import time


def DCTpower(c):
    return 20*np.log10(np.abs(c))

def Dksparse(Kmax):
    Dk = np.zeros((Kmax, Kmax), dtype=int)
    arr_k_idxs = np.repeat(np.array(np.arange(Kmax).reshape(-1, 1)), Kmax, axis=1)
    arr_j_idxs = np.repeat(np.array(np.arange(Kmax).reshape(1, -1)), Kmax, axis=0)
    idx_dist_arr = np.abs(arr_k_idxs - arr_j_idxs)
    Dk[np.logical_and(np.logical_and(2 < arr_k_idxs, arr_k_idxs < 282), idx_dist_arr == 2)] = 1
    Dk[np.logical_and(np.logical_and(282 <= arr_k_idxs, arr_k_idxs < 570), np.logical_and(2 <= idx_dist_arr, idx_dist_arr <= 13))] = 1
    Dk[np.logical_and(np.logical_and(570 <= arr_k_idxs, arr_k_idxs < 1152), np.logical_and(2 <= idx_dist_arr, idx_dist_arr <= 27))] = 1
    return Dk

def STinit(c, D):
    c_power = DCTpower(c)
    D = D.astype(bool)
    ST_arr = np.zeros((D.shape[0], )).astype(bool)
    for k in range(3, D.shape[0]):
        idx_t1 = np.array(np.clip([k - 1, k + 1], 0, D.shape[0] - 1))
        neibs_idxs = np.argwhere(D[k, :], ).reshape(-1, )
        idx_t2 = np.array(np.clip(np.append(k-neibs_idxs, k+neibs_idxs, axis=0).reshape(-1, ), 0, D.shape[0]-1))
        idx_t1, idx_t2 = idx_t1[idx_t1!=k], idx_t2[idx_t2!=k]
        term1 = (c_power[k] > c_power[idx_t1]).all()
        term2 = (c_power[k] > c_power[idx_t2] + 7).all()
        ST_arr[k] = (term1 and term2)
    ST_arr = np.argwhere(ST_arr).reshape(-1, )
    return ST_arr

def MaskPower(c, ST):
    c_power = DCTpower(c)
    c_power = np.insert(c_power, [0, c_power.shape[0]], 0)
    ST = ST + 1
    c_invlog = 10**(0.1*c_power)
    power_tones = 10*np.log10(c_invlog[ST] + c_invlog[ST - 1] + c_invlog[ST + 1])
    return power_tones

def Hz2Barks(f):
    return 13*np.arctan(0.00076*f) + 3.5*np.arctan((f/7500)**2)

def STreduction(ST, c, Tq, fs=44100):
    pm_st = MaskPower(c, ST)
    ST_f = ST[pm_st >= Tq[ST]]
    pm_f = pm_st[pm_st >= Tq[ST]]
    barks = Hz2Barks(np.arange(c.shape[0])*fs/(2*c.shape[0]))
    barks_st = barks[ST_f]
    barks_diff = np.abs(barks_st.reshape(-1, 1) - barks_st.reshape(1, -1))
    ST_out = np.array([], dtype=int)
    pm_out = np.array([])
    for i in range(ST_f.shape[0]):
        barks_diff_st = barks_diff[i]
        close_st = barks_diff_st < 0.5
        if((ST_f[i] >= ST_f[close_st]).all()):
            ST_out = np.append(ST_out, ST_f[i])
            pm_out = np.append(pm_out, pm_f[i])
    return ST_out, pm_st

def SpreadFunc(ST, PM, Kmax, fs=44100):
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
    spf = SpreadFunc(ST, PM, Kmax, fs=44100)
    tm = np.zeros((Kmax + 1, ST.shape[0]))
    barks = Hz2Barks(np.arange(Kmax + 1)*fs/(2*(Kmax + 1)))
    for i in range(Kmax + 1):
        for j in range(ST.shape[0]):
            tm[i, j] = PM[j] - 0.275 * barks[j] + spf[i, j] - 6.025
    return tm

def Global_Masking_Thresholds(Ti, Tq):
    invlog_Tq = np.power(10, 0.1*Tq)
    sum_invlog_Ti = np.sum(np.power(10, 0.1*Ti), axis=1)
    Tg = 10*np.log10(invlog_Tq + sum_invlog_Ti)
    return Tg

def psycho(c, D, Tq_path='./protocol_files/Tq.npy'):
    Tq = np.load(Tq_path)[0]
    Tq[np.isnan(Tq)] = Tq[~np.isnan(Tq)].max()
    ST = STinit(c, D)
    ST_red, ST_power = STreduction(ST, c, Tq)
    tm = Masking_Thresholds(ST_red, ST_power, c.shape[0] - 1)
    Tg = Global_Masking_Thresholds(tm, Tq)
    return Tg