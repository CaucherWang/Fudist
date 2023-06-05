import os
import random
import numpy as np
import struct
from pywt import wavedec
from numba import njit


source = './data/'
datasets_map = {
    'imagenet': (6, 200),
    # 'msong': (6, 1000),
    # 'word2vec': (6, 1000),
    # 'ukbench': (8, 200),
    # 'deep': (4, 1000),
    # 'gist': (8, 1000),
    # 'glove1.2m': (8, 1000),
    # 'sift': (8, 1000),
    # 'tiny5m': (8, 1000),
    # 'uqv':(8,1000),
    # 'glove-100':(4,1000),
    # 'crawl': (6, 1000),
    # 'mnist': (8, 1000),
    # 'cifar': (8, 1000),
    # 'sun':(8, 200),
    # 'notre':(8, 200),
    # 'nuswide':(10, 200),
    # 'trevi': (8, 200)
}

def to_floats(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            a = struct.pack('f', y)
            fp.write(a)


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


@njit
def calc_approx_dist(X, Q, RealDist, dim_proportion):
    # Q: query vector
    # X_code: PQ encoded base vector
    # RealDist: real distance between Q and X
    # return: approximated distance between Q and X / RealdDist
    result = []
    D = Q.shape[1]
    for i in range(Q.shape[0]):
        for j in range(X.shape[0]):
            dist = np.sum((Q[i][:int(D*dim_proportion)] - X[j][:int(D*dim_proportion)]) ** 2)
            if RealDist[i][j] == 0:
                if dist == 0:
                    result.append(1)
            else:
                result.append(dist / RealDist[i][j])
    return result


if __name__ == "__main__":
    
    for dataset in datasets_map.keys():
        np.random.seed(0)

        sampleQuery = datasets_map[dataset][1]
        sampleBase = 10000
        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        dist_path = os.path.join(path, f'Real_Dist_{sampleBase}_{sampleQuery}.fvecs')

        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        D = X.shape[1]
        print(f"{dataset} of dimensionality {D} of cardinality {X.shape[0]}.")
        Q = read_fvecs(query_path)
        print(f"{query_path} of dimensionality {D} of cardinality {Q.shape[0]}.")
        assert D == Q.shape[1]
        
        RealDist = read_fvecs(dist_path)

        # pad length into 2^x
        dim = Q.shape[1]
        new_dim = int(2 ** np.ceil(np.log2(dim)))
        Q = np.pad(Q, ((0, 0), (0, new_dim - dim)), 'constant')
        X = np.pad(X, ((0, 0), (0, new_dim - dim)), 'constant')

        # D = X.shape[1] # read data vectors
        data_dwt_ori = wavedec(X, 'db1') # dwt of data vectors
        query_dwt_ori = wavedec(Q, 'db1') 
        data_dwt = np.concatenate(data_dwt_ori,axis=1) # concat the info of different levelsc
        query_dwt = np.concatenate(query_dwt_ori,axis=1)

        # reove padding in transfered-domain
        zero_columns = np.all(query_dwt == 0, axis=0)
        query_dwt = query_dwt[:sampleQuery, ~zero_columns]
        data_dwt = data_dwt[:sampleBase, ~zero_columns]
        
        for p in [0.2, 0.4, 0.6, 0.8, 1]:
            result = calc_approx_dist(data_dwt, query_dwt, RealDist, p)
            result_path = os.path.join(path, f'DWT_{p}_approx_dist.floats')
            print(f'proportion of dim: {p} -> average ratio: {np.mean(result)}')
            to_floats(result_path, result)

        
        # check_no_false_dismissal(query, query_dwt)