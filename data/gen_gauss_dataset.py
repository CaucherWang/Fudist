import os
import numpy as np
import struct
import time

source = './data/'


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
    return np.random.normal(size=(D, lowdim))


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


dataset = 'gauss'
query_num = 200
if __name__ == "__main__":
    
    cardinality = 1000000
    for dim in [1000, 2000, 4000]:
        np.random.seed(int(time.time()))
        X = generate_matrix(dim, cardinality)
        Q = generate_matrix(dim, query_num)
        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}{dim}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}{dim}_query.fvecs')
        
        to_fvecs(data_path, X)
        to_fvecs(query_path, Q)



        
        


