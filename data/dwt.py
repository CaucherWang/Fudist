import os
import numpy as np
import struct
from pywt import wavedec

source = './data/'
datasets = ['gist']

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


if __name__ == "__main__":
    
    for dataset in datasets:
        np.random.seed(0)
        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')

        data_dwt_path = os.path.join(path, f'DWT_{dataset}_base.fvecs')
        query_dwt_path = os.path.join(path, f'DWT_{dataset}_query.fvecs')

        print(f"Reading {dataset} from {data_path} and {query_path}.")
        data = read_fvecs(data_path)
        query = read_fvecs(query_path)
        # D = X.shape[1] # read data vectors
        data_dwt_ori = wavedec(data, 'db1') # dwt of data vectors
        query_dwt_ori = wavedec(query, 'db1') 
        data_dwt = np.concatenate(data_dwt_ori,axis=1) # concat the info of different levelsc
        query_dwt = np.concatenate(query_dwt_ori,axis=1)

        print(f'Trans complete. {query_dwt.shape[1]} Dimensions for each vector.')

        to_fvecs(data_dwt_path, data_dwt)
        to_fvecs(query_dwt_path, query_dwt)
