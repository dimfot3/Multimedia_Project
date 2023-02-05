import numpy as np


def RLE(message, K):
    """
    RLE runs run length encoding to efficiently encode 
    sequential repetitions of a symbol.

    :param message: the symbols where RLE will be executed
    :param K: the number of DCT coefficients
    :return: a Rx2 matrix R the sequence repetitions that 
    are found. In the first column there is the symbol that observed 
    and in the second column there is the number of repetitions.
    """
    encoded_message = []
    count = 0
    last_sym = message[0]
    for i in range(1, K):
        if message[i] == 0:
            count += 1
        else:
            encoded_message.append([last_sym, count])
            last_sym = message[i]
            count = 0
    encoded_message.append([last_sym, count])
    encoded_message = np.array(encoded_message, dtype='int')
    return encoded_message

def RLE_inv(run_symbols, K):
    """
    RLE_inv inverse the run length encoding.

    :param run_symbols: the Rx2 array where R is the sequences of repeated symbols. 
    In the first column there is the symbol that observed 
    and in the second column there is the number of repetitions.
    :param K: the number of DCT coefficients
    :return: the symbols before RLE
    """
    symb_index = np.array([], dtype='int')
    for element in run_symbols:
        symb_index = np.append(symb_index, element[0])
        symb_index = np.append(symb_index, np.zeros((element[1], ), dtype=int))
    symb_index = symb_index[:K]
    return symb_index