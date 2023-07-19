import os
import numpy as np
import struct
import time
from numba import njit

source = './data/'
# datasets = ['deep', 'gist', 'glove1.2m', 'msong', 'sift', 'tiny5m', 'ukbench', 'word2vec']
datasets = ['spacev100m']


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
                
def to_fvecs2(filename, data):
    print(f"Writing File - {filename}")
    dim = (np.int32)(data.shape[1])
    data = data.astype(np.float32)
    # ff = np.memmap(path, dtype='int32', shape=(len(array), dim+1), mode='w+')
    # ff[:, 0] = dim
    # ff[:, 1:] = array.view('int32')
    # del ff
    dim_array = np.array([dim] * len(data), dtype='int32')
    file_out = np.column_stack((dim_array, data.view('int32')))
    file_out.tofile(filename)
      
def fbin_read(fname: str):
    a = np.memmap(fname, dtype='int32', mode='r')
    # a = np.fromfile(fname + ".fbin", dtype='int32')
    num = a[0]
    d = a[1]
    print(f"{num} * {d}")
    return a[2:].reshape(-1, d)[:num, :].copy().view('float32')

def fbin_read_cnt(fname: str, num: int):
    a = np.memmap(fname, dtype='int32', mode='r', shape=(2,))
    # a = np.fromfile(fname + ".fbin", dtype='int32', count=2)
    d = a[1]
    assert(a[0] >= num)
    a = np.memmap(fname, dtype='float32', mode='r', shape=(num * d + 2,))
    # a = np.fromfile(fname + ".fbin", dtype='int32', count=num * d + 2)
    return a[2:].reshape(num, d)[:num, :].copy().view('float32')

def ibin_read(fname: str):
    a = np.fromfile(fname + ".ibin", dtype='int32')
    num = a[0]
    d = a[1]
    return a[2:].reshape(-1, d)[:num, :].copy()

def i8bin_read(fname: str):
    a = np.fromfile(fname, dtype='int32', count=2)
    num = a[0]
    d = a[1]
    a = np.fromfile(fname, dtype='int8')
    return a[8:].reshape(-1, d)[:num, :].copy()

def i8bin_read_cnt(fname: str, num):
    a = np.fromfile(fname, dtype='int32', count=2)
    assert(a[0] >= num)
    d = a[1]
    a = np.memmap(fname, dtype='int8', mode='r', shape=(num * d + 8,))
    # a = np.fromfile(fname, dtype='int8')
    return a[8:].reshape(-1, d)[:num, :].copy()

def fbin_write(x, path: str):
    x = x.astype('float32')
    f = open(path + ".fbin", "wb")
    n, d = x.shape
    np.array([n, d], dtype='int32').tofile(f)
    x.tofile(f)
    
def ibin_write(x, path: str):
    x = x.astype('int32')
    f = open(path, "wb")
    n, d = x.shape
    np.array([n, d], dtype='int32').tofile(f)
    x.tofile(f)

                
if __name__ == "__main__":
    
    for dataset in datasets:
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'spacev100m_base.i8bin')

        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        # X = read_fvecs(data_path)
        # X = fbin_read_cnt(data_path, 10000000)
        X = i8bin_read_cnt(data_path, 10000000)
        D = X.shape[1]
        print(f"{dataset} of dimensionality {D} of cardinality {X.shape[0]}.")
        
        # out_path = os.path.join(path, 'spacev100m_query.fvecs')
        out_path = './data/spacev1m/spacev10m_base.fvecs'
        to_fvecs2(out_path, X)