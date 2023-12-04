'''
generate ground truth nearest neighbors
'''

import time
import faiss
import struct
import os
import numpy as np
from utils import *
from multiprocessing.dummy import Pool as ThreadPool

source = './data/'
datasets = ['sift']


# source = '/home/hadoop/wzy/dataset/'
source = './data/'
datasets = ['laion1M']

if __name__ == '__main__':
    for dataset in datasets:
        # for dim in [50, 100, 150, 200, 250, 300, 500, 750, 1000, 2000, 4000]:
            # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        # query_path = os.path.join(path, f'{dataset}_query.fvecs')
        # data_path = os.path.join(path, f'{dataset}_base.fbin')
        
        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        # X = fbin_read(data_path)
        X = read_fvecs(data_path)
        D = X.shape[1]
        # X = L2_norm_dataset(X)
        print(X.shape)
                
        K = 100
        
        GT_I, GT_D = compute_GT_CPU(X, X, K)
        print(GT_I.shape)
        
        gt_path = os.path.join(path, f'{dataset}_self_groundtruth.ivecs')
        gt_path2 = os.path.join(path, f'{dataset}_self_groundtruth_dist.fvecs')
                    
        write_ivecs(gt_path, GT_I)
        write_ivecs(gt_path2, GT_D)
