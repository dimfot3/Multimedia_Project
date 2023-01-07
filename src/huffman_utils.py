import numpy as np


class Node:
    left_node = None
    right_node = None
    freq = None
    symbol = None
    huff_code = ''
    def __init__(self, symbol, freq, huff_code=''):
        self.symbol = symbol
        self.freq = freq
        self.huff_code = huff_code

def rle_to_arr1d(data):
    arr1d = np.zeros((data.shape[0],), dtype='int')
    rle_symbs = np.unique(data, axis=0)
    for i in range(rle_symbs.shape[0]):
        arr1d[np.logical_and(data[:, 0] == rle_symbs[i, 0], data[:, 1] == rle_symbs[i, 1])] = i
    return arr1d, np.unique(data, axis=0)

def arr1d_to_rle(arr1d, unique_dict):
    return unique_dict[arr1d]

def make_tree(array, freqs):
    # create the tree
    node_arr = np.array([Node(np.array([array[i]]), freqs[i]) for i in range(array.shape[0])], dtype=Node)
    for i in range(array.shape[0] - 1):
        node_arr = sorted(node_arr, key=lambda node: node.freq)
        new_node = Node(np.append(node_arr[0].symbol, node_arr[1].symbol, axis=0), node_arr[0].freq + node_arr[1].freq)
        new_node.left_node = node_arr[0]
        new_node.left_node.huff_code += '0'
        new_node.right_node = node_arr[1]
        new_node.right_node.huff_code += '1'
        node_arr = node_arr[1:]
        node_arr[0] = new_node
    root = node_arr[0]
    return root

def make_bits_dictionary(root, arr1unique):
    dict_b = {k:"" for k in arr1unique}
    for key in arr1unique:
        curr = root
        code = ''
        while True:
            code += curr.huff_code
            if (curr.left_node == None) and (curr.right_node == None):
                break
            curr = curr.left_node if key in curr.left_node.symbol else curr.right_node
        dict_b[key] = code
    return dict_b

def print2DUtil(root, space):
    # Base case
    if (root == None):
        return
    # Increase distance between levels
    space += 20
    # Process right child first
    print2DUtil(root.right_node, space)
    # Print current node after space
    # count
    print()
    for i in range(20, space):
        print(end=" ")
    print(root.symbol, root.huff_code)
    # Process left child
    print2DUtil(root.left_node, space)

def huff(run_symbols):
    # rle to 1d
    arr1d, unique_dict = rle_to_arr1d(run_symbols)
    # calculation of frequencies
    freqs = np.bincount(arr1d) / arr1d.shape[0]
    arr1unique = np.unique(arr1d)
    root = make_tree(arr1unique, freqs)
    dict_bits = make_bits_dictionary(root, arr1unique)
    bitstream = ''
    for i in range(arr1d.shape[0]):
        bitstream = bitstream + dict_bits[arr1d[i]]
    if(unique_dict.shape[0] == 1):
        bitstream = '0'
    return bitstream, np.append(unique_dict, np.array(freqs).reshape(-1, 1), axis=1)


def ihuff(frame_stream, frame_symbol_prob):
    if frame_symbol_prob.shape[0] == 1:
        return np.array([0, 1152]).reshape(-1, 2)
    # calculation of frequencies
    freqs = frame_symbol_prob[:, 2]
    root = make_tree(np.arange(frame_symbol_prob.shape[0]), freqs)
    #print2DUtil(root, 0)
    symbs_1d = np.array([], dtype='int')
    curr = root
    for i in range(len(frame_stream)):
        if frame_stream[i] == '0':
            curr = curr.left_node
        else:
            curr = curr.right_node
        if(curr.left_node == None) and (curr.right_node == None):
            symbs_1d = np.append(symbs_1d, curr.symbol)
            curr = root
    return frame_symbol_prob[symbs_1d, :2].astype('int')