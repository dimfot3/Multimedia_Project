import numpy as np
import subband_utils
from scipy.io import wavfile
from frameDCT import frameDCT, iframeDCT
from psychoacousticUtils import *
from quantizer_utils import *
from rle_utils import RLE, RLE_inv
from huffman_utils import huff, ihuff
from tqdm import tqdm
from bitstring import BitArray


def MP3codec(wavin, h, M, N):
    """
    MP3codec is the core of MP3 implementation. It is a 
    simple format that only implements the subband filtering.
    It encodes and decodes the data.

    :param wavin: the path the sound file
    :param h: the prototype filter response
    :param M: the number of filters
    :param N: the length of each subband
    :return: the decoded sound and the encoded frames as KxNxM array where K 
    the number of frames
    """
    # MP3 encoding
    #wavin = np.append(wavin, np.zeros((M * N - wavin.shape[0] % (M*N), )), axis=0)
    H = subband_utils.make_mp3_analysisfb(h, M)
    data_len = wavin.shape[0]
    data_padd = np.append(wavin, np.zeros((M*N, )), axis=0)
    x_buff = np.zeros((M * (N-1) + H.shape[0]))
    Dk_mat = Dksparse(M*N)
    output_file = open('./outputs/bitstream.bin', 'wb')
    add_info = {'scale_arr': [], 'B_arr': [], 'B_per_frame': [], 'huff_table': [], 'B_per_frame': []}
    for i in tqdm(range(data_len // (M*N)), desc='MP3 encoding...'):
        x_buff = data_padd[0: M * (N-1) + H.shape[0]]                       # read frame
        y_frame = subband_utils.frame_sub_analysis(x_buff, H, N)            # frame analysis to subbands
        data_padd = np.roll(data_padd, - M * N)                             # shift buffer for next frame
        frame_coef = frameDCT(y_frame)                                      # DCT to buffer
        Tg = psycho(frame_coef, Dk_mat) - 15                                # find Tg from psycoacoustic model
        symb_index, SF, B = all_bands_quantizer(frame_coef, Tg)             # quantize all bands
        rle_out = RLE(symb_index, M*N)                                      # RLE
        bts, table = huff(rle_out)                                          # huffman
        add_info['scale_arr'].append(SF)
        add_info['B_arr'].append(B)
        add_info['huff_table'].append(table)
        add_info['B_per_frame'].append(len(bts))
        save_bitstream(output_file, bts)                                    # save the frame's bitstream
    output_file.close()
    np.save("./outputs/add_info.npy", add_info)                             # save the additional info (scale factors, huffman table, Bits of each critical band)
    
    # MP3 decoding
    f = open('./outputs/bitstream.bin', 'rb')
    add_info = np.load("./outputs/add_info.npy", allow_pickle=True).tolist()
    n_frames = len(add_info['B_per_frame'])
    Y_tot = []
    for frame_i in range(n_frames):
        bits_to_read = add_info['B_per_frame'][frame_i]
        bitstream = bits2a(f.read((bits_to_read + np.sign((bits_to_read % 8)) * (8 - bits_to_read % 8)) // 8))
        rle_symbs = ihuff(bitstream, add_info['huff_table'][frame_i])
        quant_symbs = RLE_inv(rle_symbs, M*N)
        xh = all_bands_dequantizer(quant_symbs, add_info['B_arr'][frame_i], add_info['scale_arr'][frame_i])
        Y_tot.append(iframeDCT(xh))

    G = subband_utils.make_mp3_synthesisfb(h, M)
    x_hat = np.zeros((n_frames*M*N, ))
    y_frame_buff = np.zeros((N + G.shape[0] // M, M))
    Y_arr = np.vstack(Y_tot)
    Y_arr = np.append(Y_arr, np.zeros((G.shape[0] // M, M)), axis=0)
    for fr_i in range(0, n_frames):
        y_frame_buff = subband_utils.idonothing(Y_arr[0: N + G.shape[0] // M])
        x_hat[fr_i*M*N:(fr_i+1)*M*N] = subband_utils.frame_sub_synthesis(y_frame_buff, G)
        Y_arr = np.roll(Y_arr, -N, axis=0)
    return x_hat

def MP3_cod(wavin, h, M, N, output_stream='./outputs/bitstream.bin', output_addinfo='./outputs/add_info.npy'):
    H = subband_utils.make_mp3_analysisfb(h, M)
    data_len = wavin.shape[0]
    data_padd = np.append(wavin, np.zeros((M*N, )), axis=0)
    x_buff = np.zeros((M * (N-1) + H.shape[0]))
    Dk_mat = Dksparse(M*N)
    output_file = open(output_stream, 'wb')
    add_info = {'scale_arr': [], 'B_arr': [], 'B_per_frame': [], 'huff_table': [], 'B_per_frame': []}
    for i in tqdm(range(data_len // (M*N)), desc='MP3 encoding...'):
        x_buff = data_padd[0: M * (N-1) + H.shape[0]]                   # read frame
        y_frame = subband_utils.frame_sub_analysis(x_buff, H, N)        # frame analysis to subbands
        data_padd = np.roll(data_padd, - M * N)                         # shift buffer for next frame
        frame_coef = frameDCT(y_frame)                                  # DCT to buffer
        Tg = psycho(frame_coef, Dk_mat) - 15                            # find Tg from psycoacoustic model
        symb_index, SF, B = all_bands_quantizer(frame_coef, Tg)         # quantize all bands
        rle_out = RLE(symb_index, M*N)                                  # RLE
        bts, table = huff(rle_out)                                      # huffman
        add_info['scale_arr'].append(SF)
        add_info['B_arr'].append(B)
        add_info['huff_table'].append(table)
        add_info['B_per_frame'].append(len(bts))
        save_bitstream(output_file, bts)                                 # save the frame's bitstream
    output_file.close()
    np.save(output_addinfo, add_info)                                    # save the additional info (scale factors, huffman table, Bits of each critical band)

def MP3_decod(ouput_bistream, output_add_info, h, M, N):
    f = open(ouput_bistream, 'rb')
    add_info = np.load(output_add_info, allow_pickle=True).tolist()
    n_frames = len(add_info['B_per_frame'])
    Y_tot = []
    for frame_i in range(n_frames):
        bits_to_read = add_info['B_per_frame'][frame_i]
        bitstream = bits2a(f.read((bits_to_read + np.sign((bits_to_read % 8)) * (8 - bits_to_read % 8)) // 8))      # read bits and transform them to str
        rle_symbs = ihuff(bitstream, add_info['huff_table'][frame_i])                                               # inverse huffman
        quant_symbs = RLE_inv(rle_symbs, M*N)                                                                       # inverse RLE
        xh = all_bands_dequantizer(quant_symbs, add_info['B_arr'][frame_i], add_info['scale_arr'][frame_i])         # dequantize
        Y_tot.append(iframeDCT(xh))                                                                                 # inverse DCT and store frame

    G = subband_utils.make_mp3_synthesisfb(h, M)                                                                    
    x_hat = np.zeros((n_frames*M*N, ))
    y_frame_buff = np.zeros((N + G.shape[0] // M, M))
    Y_arr = np.vstack(Y_tot)
    Y_arr = np.append(Y_arr, np.zeros((G.shape[0] // M, M)), axis=0)
    for fr_i in range(0, n_frames):
        y_frame_buff = Y_arr[0: N + G.shape[0] // M]                                                                # push frame into buffer
        x_hat[fr_i*M*N:(fr_i+1)*M*N] = subband_utils.frame_sub_synthesis(y_frame_buff, G)                           # synthesis of subbands
        Y_arr = np.roll(Y_arr, -N, axis=0)                                                                          # shift data to the next frame
    return x_hat

def save_bitstream(file, bitstream):
    bits = BitArray(bin=bitstream)
    bits.tofile(file)

def bits2a(b):
    return "".join(f"{n:08b}" for n in b)

def estimated_size(bitstream_file, add_info_file):
    return 0