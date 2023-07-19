import time
import struct
import os
import faiss
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import matplotlib.pyplot as plt
import parallel_sort

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

def read_ivecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
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


source = './data/'
dataset = 'e'
ef = 500
M = 16

if __name__ == '__main__':
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base_shuf2.fvecs')
    hubs_path = os.path.join(path, f'{dataset}_hubs2.ivecs')
    print(f"Reading hubs from {hubs_path}")
    hubs = read_ivecs(hubs_path).flatten()
    gt_path2_id = os.path.join(path, f'{dataset}_self_groundtruth.ivecs')
    gt_path2 = os.path.join(path, f'{dataset}_self_groundtruth_dist.fvecs')
    gt_D = read_fvecs(gt_path2)
    gt_I = read_ivecs(gt_path2_id)
    
    hubs_gt_D = gt_D[hubs]
    hubs_gt_I = gt_I[hubs]
    
    means = np.mean(hubs_gt_D, axis=0)
    print("hubs:", means)
    tot_means = np.mean(gt_D, axis=0)
    print("total:", tot_means)


