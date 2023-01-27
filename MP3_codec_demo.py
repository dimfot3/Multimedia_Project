import numpy as np
import sys 
sys.path.insert(0, './src/')
from src.MP3_total_pipeline import MP3codec, MP3_cod, MP3_decod, estimated_size
from scipy.io import wavfile
import matplotlib.pyplot as plt
import argparse


def main(args):
    # load the prototype response
    h_pr = np.load(args['h_proto_path'], allow_pickle=True).tolist()['h'].reshape(-1, )
    M = args['M']
    N = args['N']
    if args['mode'] == 'codec':
        sound_file = args['sound_data']
        samplerate, data = wavfile.read(sound_file)
        xhat = MP3codec(data, h_pr, M, N)
        wavfile.write(args['output_decoded_data'], samplerate, xhat.astype('int16'))
        # Evaluate compression
        output_bytes = estimated_size(args['out_bitstream_path'], args['add_info_path'])
        print(f'Initial data size (MB){data.shape[0] * 2 / (1024*1024)}\n'+
        f'Compressed data size (MB){output_bytes/(1024*1024)}\n' +
        f'Compression Ratio: {(data.shape[0] * 2) / output_bytes}\n'+
        f'Saved space (from initial): {(data.shape[0] * 2 - output_bytes)*100 / (data.shape[0] * 2)}%')
    elif args['mode'] == 'encoder':
        sound_file = args['sound_data']
        samplerate, data = wavfile.read(sound_file)
        MP3_cod(data, h_pr, M, N)
        # Evaluate compression
        output_bytes = estimated_size(args['out_bitstream_path'], args['add_info_path'])
        print(f'Initial data size (MB){data.shape[0] * 2 / (1024*1024)}\n'+
        f'Compressed data size (MB){output_bytes/(1024*1024)}\n' +
        f'Compression Ratio: {(data.shape[0] * 2) / output_bytes}\n'+
        f'Saved space (from initial): {(data.shape[0] * 2 - output_bytes)*100 / (data.shape[0] * 2)}%')
    elif args['mode'] == 'decoder':
        sound_file = args['sound_data']
        samplerate, data = wavfile.read(sound_file)
        xhat = MP3_decod(args['out_bitstream_path'], args['add_info_path'], h_pr, M, N)
        wavfile.write(args['output_decoded_data'], samplerate, xhat.astype('int16'))

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='This is the demo for MP3 like encoder custom implementation.')
    parser.add_argument('--mode', type=str,
                    help='The mode of demo to run. There are three options. The "codec" that \
                        performs encoding and decoding of a sound data. The "encoder" that \
                        perform encoding of data only and saves the bitstream and addiotional info \
                        and the  "decoder" that decodes the bitstream and additional info and saves a \
                        music sound. default: codec', default='codec')
    parser.add_argument('--h_proto_path', type=str, default='./protocol_files/h.npy',
                    help='The path to prototype filter for subband analysis. default: ./protocol_files/h.npy')
    parser.add_argument('--M', type=int,
                        help='The number of subband filter. default: 32', default=32)
    parser.add_argument('--N', type=int,
                        help='The number of samples for each subband. default: 36', default=36)
    parser.add_argument('--sound_data', default='./data/myfile.wav',
                    help='The file that will apply the encoding. default: ./data/myfile.wav')
    parser.add_argument('--out_bitstream_path', default='./outputs/bitstream.bin',
                    help='The file of the output binary file of the encoded bistream. default: ./outputs/bitstream.bin')
    parser.add_argument('--add_info_path', default='./outputs/add_info.npy',
                    help='The file of the output binary file of the additional info of encoded bitstream. default: ./outputs/add_info.npy')
    parser.add_argument('--output_decoded_data', default='./outputs/final.wav',
                    help='The output file of the decoded signal. default: ./outputs/final.wav')
    args = vars(parser.parse_args())
    main(args)
