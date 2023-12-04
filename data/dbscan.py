from sklearn.cluster import dbscan
import os
import numpy as np
source = './data/'
datasets = ['glove1.2m']

def read_fvecs(filename, c_contiguous=True):
    # fv = np.fromfile(filename, dtype=np.float32)
    fv = np.memmap(filename, dtype='float32', mode='r+')
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    # if not all(fv.view(np.int32)[:, 0] == dim):
    #     raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def ibin_write(x, path: str):
    x = x.astype('int32')
    f = open(path, "wb")
    x.tofile(f)


dataset = datasets[0]
path = os.path.join(source, dataset)
data_path = os.path.join(path, f'{dataset}_base.fvecs')
X = read_fvecs(data_path)
eps = 200
min_samples = 10
core_samples, cluster_ids = dbscan(X, eps=eps, min_samples=min_samples,  n_jobs=-1)
ibin_write(cluster_ids, os.path.join(path, f'{dataset}_dbscan_cluster_{eps}_{min_samples}.ibin'))

