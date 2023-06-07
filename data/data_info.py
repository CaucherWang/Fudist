import os
import numpy as np
import struct
import time

source = './data/'
# datasets = ['deep', 'gist', 'glove1.2m', 'msong', 'sift', 'tiny5m', 'ukbench', 'word2vec']
datasets = ['imagenet']

datasets_map = {
    'imagenet': (6, 200),
    # 'msong': (6, 1000),
    # 'word2vec': (6, 1000),
    # 'ukbench': (8, 200),
    # 'deep': (8, 1000),
    # 'gist': (8, 1000),
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
        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')


        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        Q = read_fvecs(query_path)
        print('X.shape', X.shape)
        print('Q.shape', Q.shape)
        