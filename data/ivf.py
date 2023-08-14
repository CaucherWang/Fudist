
import numpy as np
import faiss
import struct
import os

source = './'
datasets = ['gist']
# the number of clusters
K = 4096 

def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q

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

if __name__ == '__main__':

    for dataset in datasets:
        print(f"Clustering - {dataset}")
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        centroids_path = os.path.join(path, f'{dataset}_cn{K}.fvecs')

        trans_data_path = os.path.join(path, f'O{dataset}_base.fvecs')
        randomzized_cluster_path = os.path.join(path, f"O{dataset}_cn{K}.fvecs")
        transformation_path = os.path.join(path, 'O.fvecs')

        # read data vectors
        X = read_fvecs(data_path)

        D = X.shape[1]
        # P = Orthogonal(D)
        # to_fvecs(transformation_path, P)

        P = read_fvecs(transformation_path)

        XP = np.dot(X, P)
        to_fvecs(trans_data_path, XP)

        # cluster data vectors
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        to_fvecs(centroids_path, centroids)

        # randomized centroids
        centroids = np.dot(centroids, P)
        to_fvecs(randomzized_cluster_path, centroids)
