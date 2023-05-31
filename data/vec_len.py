import os
import numpy as np
import struct
import math
from numba import njit

source = './data/'
# 6, 200
# datasets = ['imagenet']
# 6, 1000
# datasets = ['msong', 'word2vec']
# 8, 200
# datasets = ['ukbench']
# 8, 1000
datasets = ['deep', 'gist', 'glove1.2m', 'sift', 'tiny5m']

datasets_map = {
    'imagenet': (6, 200),
    'msong': (6, 1000),
    'word2vec': (6, 1000),
    'ukbench': (8, 200),
    'deep': (8, 1000),
    'gist': (8, 1000),
    'glove1.2m': (8, 1000),
    'sift': (8, 1000),
    'tiny5m': (8, 1000),
}

def to_ivecs(filename: str, array: np.ndarray):
    print(f"Writing File - {filename}")
    topk = (np.int32)(array.shape[1])
    array = array.astype(np.int32)
    topk_array = np.array([topk] * len(array), dtype='int32')
    file_out = np.column_stack((topk_array, array.view('int32')))
    file_out.tofile(filename)

def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)
                
def read_floats(filename, c_contiguous=True):
    print(f"Reading File - {filename}")
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    if c_contiguous:
        fv = fv.copy()
    return fv

                
def to_floats(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            a = struct.pack('f', y)
            fp.write(a)

@njit
def compute_vec_len(X):
    '''
    X: N * D , where each row is a vector
    return: N * 1, where each row is the length of the corresponding vector
    '''
    return np.sum(X ** 2, axis=1)
    
if __name__ == "__main__":
    
    for dataset in datasets_map.keys():
        np.random.seed(0)
        
        sampleQuery = datasets_map[dataset][1]
        sampleBase = 10000
        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')

        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        D = X.shape[1]
        print(f"{dataset} of dimensionality {D} of cardinality {X.shape[0]}.")
        
        transformed_path = os.path.join(path, 'base_vec_len.floats')
        
        R = compute_vec_len(X)

        to_floats(transformed_path, R)


