# Multimedia_Project
This is a project for multimedia, subject of 9th semester of Electrical and Computer Engineering in Aristotle University of Thessaloniki. The MP3 protocol is a form of audio compression used for storing and transmitting audio files over the internet and on audio playback devices. MP3 consists of a compression algorithm that reduces the resolution of the audio without significantly affecting its quality. It is one of the most widely used audio compression protocols and is based on the ISO/IEC 11172-3 protocol.

The basic components of MP3 are a frequency analysis of the input signal and independent samples. This is followed by a controlled distortion of the signal, which is influenced by a psychoacoustic analysis that reduces the quantization accuracy in areas of the spectrum where the quantization error is not noticeable. Then we have Run Length Encoding and entropy encoding through Huffman.

 This project is a simplified implementation of MP3 codec.

## Project Structure
In the root folder there there are:
1. the data folder where the wav files are stored.
2. the protocol_files folder where the protocol coefficients are saved.
3. the src folder where the core functionalities of the project are implemented.
4. the scripts subband_filtering_demo.py, dct_demo.py, psychoacoustic_th_demo.py, quantizer_demo.py, RLE.py, huffman_demo.py, MP3_codec_demo.py are some scripts that proove the correctness of the project main functionalities.
5. the report.pdf is the report of this project.

In the src the functionalities are splitted to:
1. utils.py where basic functions are implemented. Here the given functons used in  encoder0 are saved.
2. encoder0.py where codec0, coder0 and decoder0  are implemented.
3. frameDCT.py where frameDCT and iframeDCT rutines are implemented.
4. psychoacousticUtils.py where psyco and its related functions are implemented
5. rle_utils.py whre RLE and RLE_inv are implemented.
6. huffman_utils.py where huff and ihuff rutines are implemented.
7. MP3_total_pipeline.py where the final codec1, coder1 and decoder1 that include all the MP3
utilities are implemented. 

## How to run
First of all it is recomended to run `pip install -r requirements.txt` to install the correct python packages that are used in this project.

In root folder there are some scripts that can verify correctness of basic parts of the project like the subbands, dct, psychoacoustic model, quantizer, RLE and huffman.

Except from these that are mostly demostration of correctness, there is the MP3_codec_demo.py that run the whole MP3 pipeline. Some example of 
running this script are

- sound codec (encoder/decoder) `python3 MP3_codec_demo.py --mode codec --sound_data ./data/myfile.wav --out_bitstream_path ./outputs/bitstream.bin --add_info_path ./outputs/add_info.npy --output_decoded_data ./outputs/final.wav`
- sound encoder `python3 MP3_codec_demo.py --mode encoder --sound_data ./data/myfile.wav --out_bitstream_path ./outputs/bitstream.bin --add_info_path ./outputs/add_info.npy`
- sound decoder `python3 MP3_codec_demo.py --mode decoder --sound_data ./data/myfile.wav --out_bitstream_path ./outputs/bitstream.bin --add_info_path ./outputs/add_info.npy --output_decoded_data ./outputs/final.wav`


