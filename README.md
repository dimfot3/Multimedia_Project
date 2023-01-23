# Multimedia_Project
This is a project for multimedia, subject of 9th semester of Electrical and Computer Engineering in Aristotle University of Thessaloniki.


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
In root folder there are some scripts that can verify correctness of basic parts of the project like the subbands, dct, psychoacoustic model, quantizer, RLE and huffman.

Except from these that are mostly demostration of correctness, there is the MP3_codec_demo.py that run the whole MP3 pipeline.
