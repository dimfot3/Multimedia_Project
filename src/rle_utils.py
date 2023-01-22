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
    encoded_message = np.zeros((0, 2)).astype(int)
    i = 0
    while (i <= K - 1):
        count = 1
        ch = message[i]
        j = i
        while (j < K-1):
            if (message[j] == message[j+1]):
                count = count+1
                j = j+1
            else:
                break
        encoded_message = np.append(encoded_message, np.array([ch, count]).reshape(1, 2), axis=0)
        i = j+1
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
    symb_index = np.zeros((K, ), dtype='int')
    last_idx = 0
    for i in range(run_symbols.shape[0]):
        symb_index[last_idx:last_idx+run_symbols[i, 1]] = run_symbols[i, 0]
        last_idx = last_idx + run_symbols[i, 1]
    return symb_index