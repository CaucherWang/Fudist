import numpy as np
import os
import struct
import matplotlib.pyplot as plt

def fbin_read(fname: str):
    a = np.memmap(fname, dtype='int32', mode='r')
    # a = np.fromfile(fname + ".fbin", dtype='int32')
    num = a[0]
    d = a[1]
    print(f"{num} * {d}")
    return a[2:].reshape(-1, d)[:num, :].copy().view('float32')

def fbin_write(x, path: str):
    x = x.astype('float32')
    f = open(path, "wb")
    n, d = x.shape
    np.array([n, d], dtype='int32').tofile(f)
    x.tofile(f)
    
def rand_select(nq, nx):
    # randomly select nq numbers from [0,nx)
    return np.random.choice(nx, nq, replace=False)

def ibin_write(x, path: str):
    x = x.astype('int32')
    f = open(path, "wb")
    x.tofile(f)


def ibin_read(fname: str):
    a = np.fromfile(fname , dtype='int32')
    num = a[0]
    d = a[1]
    return a[2:].reshape(-1, d)[:num, :].copy()

source = '/home/hadoop/wzy/dataset/'
datasets = ['yandex-text2image']
if __name__ == '__main__':
    for dataset in datasets:
        dir = os.path.join(source, dataset)
        id_query_file = os.path.join(dir, f'base.skip.1M.1k.fbin')
        data_file = os.path.join(dir, f'base.1M.fbin')
        ood_query_file = os.path.join(dir, f'query.public.1k.fbin')
        id_gt_file = os.path.join(dir, f'groundtruth.base.1M.base.1k.ibin')
        ood_gt_file = os.path.join(dir, f'groundtruth.base.1M.query.1k.ibin')
        X = fbin_read(data_file)
        id_query = fbin_read(id_query_file)
        ood_query = fbin_read(ood_query_file)
        id_gt = ibin_read(id_gt_file)
        ood_gt = ibin_read(ood_gt_file)
                
        X_id_gt = X[id_gt]
        X_ood_gt = X[ood_gt]
        
        nq = id_query.shape[0]
        
        id_dists = []
        ood_dists = []
        for i in range(nq):
            idd = id_query[i] @ X_id_gt[i].T
            oodd = ood_query[i] @ X_ood_gt[i].T
            id_dists.append([idd[0] - idd[9], idd[0] - idd[19], idd[0] - idd[49], idd[0] - idd[99]])
            ood_dists.append([oodd[0] - oodd[9], oodd[0] - oodd[19], oodd[0] - oodd[49], oodd[0] - oodd[99]])
        
        id_dists = np.array(id_dists)
        ood_dists = np.array(ood_dists)
        
        # compute the mean values of columns in id_dists and ood_dists
        id_means = np.mean(id_dists, axis=0)
        ood_means = np.mean(ood_dists, axis=0)
 
        print(id_means)
        print(ood_means)