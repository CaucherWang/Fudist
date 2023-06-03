import h5py
import numpy as np
import struct
import os

def read_hdf5_dat(filepath):
    '''
    Read data from hdf5 file
    '''
    with h5py.File(filepath, 'r') as f:
        for name in f:
            if isinstance(f[name], h5py.Dataset):
                print(name)
        data = f['dataset'][:]
        query = f['query'][:]
        gt = f['groundtruth'][:]
        return data, query, gt
    
def to_ivecs(filename: str, array: np.ndarray):
    print(f"Writing File - {filename}")
    topk = (np.int32)(array.shape[1])
    array = array.astype(np.int32)
    topk_array = np.array([topk] * len(array), dtype='int32')
    file_out = np.column_stack((topk_array, array.view('int32')))
    file_out.tofile(filename)
    

def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)
           
           

source = './data/'
dataset = 'sun'

if __name__ == '__main__':
    path = os.path.join(source, dataset)

    X,Q,GT = read_hdf5_dat(os.path.join(path, f'{dataset}.hdf5'))
    
    to_fvecs(os.path.join(path, f'{dataset}_base.fvecs'), X)
    to_fvecs(os.path.join(path, f'{dataset}_query.fvecs'), Q)
    to_ivecs(os.path.join(path, f'{dataset}_groundtruth.ivecs'), GT)