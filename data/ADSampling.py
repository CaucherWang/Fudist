import os
import numpy as np
import struct
from numba import njit
import math

source = './data/'
datasets = ['imagenet', 'ukbench', 'word2vec']
datasets_map = {
    # 'imagenet': (6, 200),
    # 'msong': (6, 1000),
    # 'word2vec': (6, 1000),
    # 'ukbench': (8, 200),
    'deep': (8, 1000),
    'gist': (8, 1000),
    'glove1.2m': (8, 1000),
    'sift': (8, 1000),
    # 'tiny5m': (8, 1000),
}

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

def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q

def to_floats(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            a = struct.pack('f', y)
            fp.write(a)
            

@njit
def calc_approx_dist(X, Q, RealDist, r):
    # Q: query vector
    # X_code: PQ encoded base vector
    # RealDist: real distance between Q and X
    # return: approximated distance between Q and X / RealdDist
    result = []
    for i in range(Q.shape[0]):
        for j in range(X.shape[0]):
            dist = np.sum((Q[i] - X[j]) ** 2) / r
            if RealDist[i][j] == 0:
                if dist == 0:
                    result.append(1)
            else:        
                result.append(dist / RealDist[i][j])
    return result

def ratio(D, i, epsilon0):
    if i == D:
        return 1.0
    return 1.0 * i / D * (1.0 + epsilon0 / math.sqrt(i)) * (1.0 + epsilon0 / math.sqrt(i))

if __name__ == "__main__":
    
    for dataset in datasets_map.keys():
        np.random.seed(0)
        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')

        # read data vectors
        # print(f"Reading {dataset} from {data_path}.")
        # X = read_fvecs(data_path)
        # D = X.shape[1]

        # # generate random orthogonal matrix, store it and apply it
        # print(f"Randomizing {dataset} of dimensionality {D}.")
        # P = Orthogonal(D)
        # XP = np.dot(X, P)

        projection_path = os.path.join(path, 'O.fvecs')
        transformed_path = os.path.join(path, f'O{dataset}_base.fvecs')

        # to_fvecs(projection_path, P)
        # to_fvecs(transformed_path, XP)

        sampleQuery = datasets_map[dataset][1]
        sampleBase = 10000
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        dist_path = os.path.join(path, f'Real_Dist_{sampleBase}_{sampleQuery}.fvecs')
        sample = 0.2
        U = read_fvecs(projection_path)
        lowdim = int(U.shape[0] * sample)
        r = ratio(U.shape[0], lowdim, 2.1)
        
        X = read_fvecs(transformed_path)[:sampleBase, :lowdim]
        Q = read_fvecs(query_path)[:sampleQuery]
        Q = np.dot(Q, U)[ : , :lowdim]
        RealDist = read_fvecs(dist_path)
        result = calc_approx_dist(X, Q, RealDist, r)
        result = np.array(result)
        result = np.sort(result)
        result_path = os.path.join(path, f'ADS_{sample}_approx_dist.floats')
        to_floats(result_path, result)
        