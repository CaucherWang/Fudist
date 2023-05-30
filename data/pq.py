import os
import numpy as np
import struct
import nanopq

source = './data/'
# 6, 200
# datasets = ['imagenet']
# 6, 1000
# datasets = ['msong', 'word2vec']
# 8, 200
# datasets = ['ukbench']
# 8, 1000
datasets = ['deep', 'gist', 'glove1.2m', 'sift', 'tiny5m']

datsets_map = {
    'imagenet': (6, 200),
    'msong': (6, 1000),
    'word2vec': (6, 1000),
    'ukbench': (8, 200),
    'deep': (8, 1000),
    'gist': (8, 1000),
    'glove1.2m': (8, 1000),
    'sift': (8, 1000),
    'tiny5m': (8, 1000),
}

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
                
def read_floats(filename, c_contiguous=True):
    print(f"Reading File - {filename}")
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    if c_contiguous:
        fv = fv.copy()
    return fv

                
def to_floats(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            a = struct.pack('f', y)
            fp.write(a)

def to_fdat(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        d1 = struct.pack('I', data.shape[0])
        d2 = struct.pack('I', data.shape[1])
        d3 = struct.pack('I', data.shape[2])
        fp.write(d1)
        fp.write(d2)
        fp.write(d3)
        for z in data:
            for y in z:
                for x in y:
                    a = struct.pack('f', x)
                    fp.write(a)


def calc_approx_dist(Q, X_code, pq, RealDist):
    # Q: query vector
    # X_code: PQ encoded base vector
    # RealDist: real distance between Q and X
    # return: approximated distance between Q and X / RealdDist
    result = []
    for i in range(Q.shape[0]):
        dists = pq.dtable(query = Q[i]).adist(codes = X_code)
        for j in range(len(dists)): 
            if RealDist[i][j] == 0:
                if dists[j] == 0:
                    result.append(1)
            else:        
                result.append(dists[j] / RealDist[i][j])
    return result

if __name__ == "__main__":
    
    for dataset in datasets:
        np.random.seed(0)
        
        sampleQuery = datsets_map[dataset][1]
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
        
        M = 8
        Ks = 256
        M = datsets_map[dataset][0]
        
        pq = nanopq.PQ(M=M, Ks=Ks, verbose=True)
        pq.fit(vecs=X, iter=20, seed=123)
        X_code = pq.encode(vecs=X[:sampleBase])
        print(X_code.shape)
        
        result = calc_approx_dist(Q[:sampleQuery], X_code, pq, RealDist)
        print(len(result))
        # sort the result asc
        result = np.array(result)
        result = np.sort(result)
        # 90% percent of result
        print(result[int(len(result) * 0.9)])
        
        result_path = os.path.join(path, f'PQ_{M}_{Ks}_approx_dist.floats')
        to_floats(result_path, result)

        # projection_path = os.path.join(path, f'PQ_codebook_{M}_{Ks}.fdat')
        # transformed_path = os.path.join(path, f'PQ_{M}_{Ks}_{dataset}_base.ivecs')

        # to_fdat(projection_path, pq.codewords)
        # to_ivecs(transformed_path, X_code)


