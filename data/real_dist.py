import os
import numpy as np
import struct
import nanopq
from numba import njit

source = './data/'
datasets = ['trevi']
# datasets = ['deep', 'glove1.2m']
# datasets = ['word2vec', 'imagenet']
# datasets = ['msong', 'sift', 'tiny5m', 'ukbench', 'word2vec']

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


@njit
def calc_L2_dist(Q, D):
    '''
    calc Euclidean distance between each row of Q and each row of D
    '''
    # Q: N x D  D: M x D
    # initialize the result matrix (N x M)
    result = np.zeros((Q.shape[0], D.shape[0]))
    # iterate over rows of Q
    for i in range(Q.shape[0]):
        for j in range(D.shape[0]):
            # calculate the Euclidean distance
            result[i, j] = np.sum((Q[i, :] - D[j, :]) ** 2)
    return result


if __name__ == "__main__":
    
    for dataset in datasets:
        np.random.seed(0)
        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        
        X_sample = 10000
        Q_sample = 1000

        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        D = X.shape[1]
        print(f"{dataset} of dimensionality {D} of cardinality {X.shape[0]}.")
        
        Q = read_fvecs(query_path)
        D_ = Q.shape[1]
        print(f"{dataset} of dimensionality {D_} of cardinality {Q.shape[0]}.")
        X_sample = min(X_sample, X.shape[0])
        Q_sample = min(Q_sample, Q.shape[0])
        
        result = calc_L2_dist(Q[:Q_sample], X[:X_sample])

        dist_path = os.path.join(path, f'Real_Dist_{X_sample}_{Q_sample}.fvecs')

        to_fvecs(dist_path, result)
