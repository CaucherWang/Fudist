import os
import numpy as np
import struct
import time

source = './data/'
# datasets = ['deep', 'gist', 'glove1.2m', 'msong', 'sift', 'tiny5m', 'ukbench', 'word2vec']
datasets = ['gist']

datasets_map = {
    # 'imagenet': (6, 200),
    # 'msong': (6, 1000),
    # 'word2vec': (6, 1000),
    # 'ukbench': (8, 200),
    # 'deep': (8, 1000),
    'gist': (8, 1000),
    # 'glove1.2m': (8, 1000),
    # 'sift': (8, 1000),
    # 'tiny5m': (8, 1000),
    # 'uqv':(8,1000),
    # 'glove-100':(4,1000),
    # 'crawl': (6, 1000),
    # 'enron': (8, 1000)
    # 'mnist': (8, 1000),
    # 'cifar': (8, 1000),
    # 'sun':(8, 200),
    # 'notre':(8, 200),
    # 'nuswide':(4, 200),
    # 'trevi': (8, 200)


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

def to_floats(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            a = struct.pack('f', y)
            fp.write(a)


def generate_matrix(lowdim, D):
    return np.random.normal(size=(lowdim, D))


def calc_approx_dist(XP, XQ, RealDist):
    # return: approximated distance between Q and X / RealdDist
    result = []
    for i in range(XQ.shape[0]):
        for j in range(XP.shape[0]):
            dist = np.sum((XQ[i] - XP[j]) ** 2)
            if RealDist[i][j] == 0:
                if dist == 0:
                    result.append(1)
            else:        
                result.append(dist / RealDist[i][j])
    return result


if __name__ == "__main__":
    
    for dataset in datasets_map.keys():
        np.random.seed(int(time.time()))
        
        sampleQuery = datasets_map[dataset][1]
        sampleBase = 10000
        lowdim = 64

        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')


        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        D = X.shape[1]
        print(f"{dataset} of dimensionality {D} of cardinality {X.shape[0]}.")
        
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        dist_path = os.path.join(path, f'Real_Dist_{sampleBase}_{sampleQuery}.fvecs')
        Q = read_fvecs(query_path)
        print(f"{query_path} of dimensionality {D} of cardinality {Q.shape[0]}.")
        assert D == Q.shape[1]
        
        RealDist = read_fvecs(dist_path)
        sampleQuery = RealDist.shape[0]
        sampleBase = RealDist.shape[1]

        
        # generate random orthogonal matrix, store it and apply it
        
        # print(f"LSH {dataset} of dimensionality {D}.")
        # P = generate_matrix(lowdim, D)
        # XP = np.dot(X, P.T)
        
        projection_path = os.path.join(path, f'LSH{lowdim}.fvecs')
        transformed_path = os.path.join(path, f'LSH{lowdim}_{dataset}_base.fvecs')

        # to_fvecs(projection_path, P.T)
        # to_fvecs(transformed_path, XP)
        
        P = read_fvecs(os.path.join(path, f'LSH{lowdim}.fvecs'))
        XP = np.dot(X[:sampleBase], P)
        XQ = np.dot(Q[:sampleQuery], P)
        
        result = calc_approx_dist(XP, XQ, RealDist)
        print(len(result))
        # sort the result asc
        result = np.array(result)
        result = np.sort(result)
        # 90% percent of result
        print(result[int(len(result) * 0.9)])
        
        result_path = os.path.join(path, f'LSH_{lowdim}_approx_dist.floats')
        to_floats(result_path, result)

        
        


