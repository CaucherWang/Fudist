import os
import random
import numpy as np
import struct
from pywt import wavedec

source = './data/'
datasets = ['enron', 'glove-100', 'glove1.2m', 'imagenet', 'msong', 'mnist', 'nuswide', 'notre', 'sift', 'sun', 'tiny5m', 'trevi', 'ukbench', 'uqv', 'word2vec']
datasets  = ['mnist']
datasets  = ['gauss50', 'gauss100', 'gauss150', 'gauss200', 'gauss250', 'gauss300', 'gauss500', 'gauss1000', 'gauss2000', 'gauss4000']
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


def ed(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def check_no_false_dismissal(data, data_dwt):
    print("Checking property.")
    cnt = 0
    for _ in range(1000):
        # print(_)
        idx1 = random.randint(0,len(data)-1)
        idx2 = random.randint(0,len(data)-1)
        dist1 = ed(data[idx1], data[idx2])
        dist2 = ed(data_dwt[idx1], data_dwt[idx2])
        if abs(dist1-dist2) > 1e-5:
            cnt += 1
    print(f"!!!!!!!!!!!!!Total {cnt} errors.!!!!!!!!!!!!!")
    

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

        # pad length into 2^x
        dim = query.shape[1]
        new_dim = int(2 ** np.ceil(np.log2(dim)))
        query = np.pad(query, ((0, 0), (0, new_dim - dim)), 'constant')
        data = np.pad(data, ((0, 0), (0, new_dim - dim)), 'constant')

        # D = X.shape[1] # read data vectors
        data_dwt_ori = wavedec(data, 'db1') # dwt of data vectors
        query_dwt_ori = wavedec(query, 'db1') 
        data_dwt = np.concatenate(data_dwt_ori,axis=1) # concat the info of different levelsc
        query_dwt = np.concatenate(query_dwt_ori,axis=1)

        # reove padding in transfered-domain
        zero_columns = np.all(query_dwt == 0, axis=0)
        query_dwt = query_dwt[:, ~zero_columns]
        data_dwt = data_dwt[:, ~zero_columns]
        
        check_no_false_dismissal(query, query_dwt)

        print(f'Trans complete. {query_dwt.shape[1]} Dimensions for each vector.')

        to_fvecs(data_dwt_path, data_dwt)
        to_fvecs(query_dwt_path, query_dwt)
