import os
import numpy as np
import struct
from numba import njit

source = './data/'
datasets = ['gist']
datasets_map = {
    # 'imagenet': (0, 200),
    # 'msong': (42, 1000),
    # 'word2vec': (30, 1000),
    # 'ukbench': (0, 200),
    # 'deep': (16, 1000),
    # 'gist': (96, 1000),
    # 'glove1.2m': (20, 1000),
    'sift': (16, 1000),
    # 'tiny5m': (24, 1000),
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


# @njit
def paa_transform(X, w):
    # X is a numpy array where each row is a time series
    # w is the number of segments
    
    n, m = X.shape
    X_paa = np.zeros((n, w))
    
    for i in range(w):
        start = int(i * m / w)
        end = int((i + 1) * m / w)
        X_paa[:, i] = np.mean(X[:, start:end], axis=1)
    
    return X_paa

def compute_mean(matrix):
    return np.mean(matrix, axis=0)

def sort_with_positions(arr):
    # create a list of tuples where the first element is the value
    # and the second element is the original position in the array
    arr_with_positions = [(val, i) for i, val in enumerate(arr)]

    # sort the list of tuples by the first element (the value)
    sorted_arr_with_positions = sorted(arr_with_positions)

    # create two separate arrays, one for the sorted values and one for the original positions
    sorted_arr = [val for val, _ in sorted_arr_with_positions]
    positions = [pos for _, pos in sorted_arr_with_positions]

    return sorted_arr, positions

def to_floats(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            a = struct.pack('f', y)
            fp.write(a)

@njit
def calc_approx_dist(X, Q, RealDist):
    # Q: query vector
    # X_code: PQ encoded base vector
    # RealDist: real distance between Q and X
    # return: approximated distance between Q and X / RealdDist
    result = []
    for i in range(Q.shape[0]):
        for j in range(X.shape[0]):
            dist = np.sum((Q[i] - X[j]) ** 2)
            if RealDist[i][j] == 0:
                if dist == 0:
                    result.append(1)
            else:        
                result.append(dist / RealDist[i][j])
    return result

if __name__ == "__main__":
    
    for dataset in datasets_map.keys():
        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        
        w = datasets_map[dataset][0]

        # read data vectors
        # print(f"Reading {dataset} from {data_path}.")
        # X = read_fvecs(data_path)
        # D = X.shape[1]
        
        # print(f"PAA {dataset} of dimensionality {D}.")
        # PAA = paa_transform(X, w)

        transformed_path = os.path.join(path, f'PAA_{w}_{dataset}_base.fvecs')
        # to_fvecs(transformed_path, PAA)
        
        
        sampleQuery = datasets_map[dataset][1]
        sampleBase = 10000
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        dist_path = os.path.join(path, f'Real_Dist_{sampleBase}_{sampleQuery}.fvecs')
        X = read_fvecs(transformed_path)[:sampleBase]
        Q = read_fvecs(query_path)[:sampleQuery]
        Q = paa_transform(Q, w)
        result = calc_approx_dist(X, Q, read_fvecs(dist_path))
        result = np.array(result)
        result = np.sort(result)
        result_path = os.path.join(path, f'PAA_{w}_approx_dist.floats')
        to_floats(result_path, result)
        
