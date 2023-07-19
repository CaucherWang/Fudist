import numpy as np
import os
import struct

def fbin_read(fname: str):
    a = np.memmap(fname, dtype='int32', mode='r')
    # a = np.fromfile(fname + ".fbin", dtype='int32')
    num = a[0]
    d = a[1]
    print(f"{num} * {d}")
    return a[2:].reshape(-1, d)[:num, :].copy().view('float32')

def fbin_write(x, path: str):
    x = x.astype('float32')
    f = open(path, "wb")
    n, d = x.shape
    np.array([n, d], dtype='int32').tofile(f)
    x.tofile(f)
    
def rand_select(nq, nx):
    # randomly select nq numbers from [0,nx)
    return np.random.choice(nx, nq, replace=False)

def ibin_read(fname: str):
    a = np.fromfile(fname + ".ibin", dtype='int32')
    num = a[0]
    d = a[1]
    return a[2:].reshape(-1, d)[:num, :].copy()

def ibin_write(x, path: str):
    print(f"Writing File - {path}")
    x = x.astype('int32')
    f = open(path, "wb")
    x.tofile(f)
    f.close()

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

def to_ivecs(filename: str, array: np.ndarray):
    print(f"Writing File - {filename}")
    topk = (np.int32)(array.shape[1])
    array = array.astype(np.int32)
    topk_array = np.array([topk] * len(array), dtype='int32')
    file_out = np.column_stack((topk_array, array.view('int32')))
    file_out.tofile(filename)

source = './data/'
datasets = ['deep']
if __name__ == '__main__':
    for dataset in datasets:
        dir = os.path.join(source, dataset)
        data_file = os.path.join(dir, f'{dataset}_base.fvecs')
        shuf_data_file = os.path.join(dir, f'{dataset}_base_shuf2.fvecs')
        pos_file = os.path.join(dir, f'{dataset}_pos2.ibin')
        X = read_fvecs(data_file)
        nx = X.shape[0]
        
        # Get the number of rows in the matrix
        rows = X.shape[0]
        # Create an array of indices representing the original positions
        original_positions = np.arange(rows)
        # Shuffle the original positions
        np.random.shuffle(original_positions)
        # Create a shuffled matrix using the shuffled positions
        X_new = X[original_positions]        
        
        print(X_new.shape)
        to_fvecs2(shuf_data_file, X_new)
        # to_ivecs(pos_file, original_positions)
        ibin_write(original_positions, pos_file)
        