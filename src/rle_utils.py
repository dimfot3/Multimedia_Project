import numpy as np

def RLE(message, K):
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
    symb_index = np.zeros((K, ), dtype='int')
    last_idx = 0
    for i in range(run_symbols.shape[0]):
        symb_index[last_idx:last_idx+run_symbols[i, 1]] = run_symbols[i, 0]
        last_idx = last_idx+run_symbols[i, 1]
    return symb_index