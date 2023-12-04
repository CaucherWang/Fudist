'''
generate ground truth nearest neighbors
'''

import os
import numpy as np
from utils import *
source = './data/'
datasets = ['deep']

if __name__ == '__main__':
    for dataset in datasets:
        # for dim in [50, 100, 150, 200, 250, 300, 500, 750, 1000, 2000, 4000]:
            # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        # data_path = os.path.join(path, f'{dataset}_base.fbin')
        # data_path = os.path.join(path, f'{dataset}_sample_query_smallest.fbin')
        # query_path = os.path.join(path, f'{dataset}_query.fbin')
        # query_path = os.path.join(path, f'{dataset}_base.fbin')

        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        # X = read_fbin(data_path)
        D = X.shape[1]
        print(X.shape)
        # X = L2_norm_dataset(X)
        
        # # read data vectors
        print(f"Reading {dataset} from {query_path}.")
        Q = read_fvecs(query_path)
        # Q = read_fbin(query_path)[:2000]
        # norms = np.linalg.norm(Q, axis=1)
        QD = Q.shape[1]
        print(Q.shape)
        # Q = L2_norm_dataset(Q)
        
        K = 50000
        
        GT_I, GT_D = compute_GT_CPU(X, Q, K)
        print(GT_I.shape)
        
        # gt = read_ibin(os.path.join(path, f'groundtruth.img_0.1M.text_0.2k.ibin'))
        # for i in range(gt.shape[0]):
        #     gt_i = gt[i][:100]
        #     gt_i_set = set(gt_i)
        #     GT_I_i = GT_I[i]
        #     GT_I_set = set(GT_I_i)
        #     # compute the lenghth of common part
        #     overlap = gt_i_set & GT_I_set
        #     # overlap_raw = np.intersect1d(gt_i_set, GT_I_set)
        #     # overlap = overlap_raw[0]
        #     if len(overlap) < 100:
        #         print(i, len(overlap))
        # gt_path = os.path.join(path, f'{dataset}_groundtruth.ivecs')
        # gt_d_path = os.path.join(path, f'{dataset}_groundtruth_dist.fvecs')
        gt_path = os.path.join(path, f'{dataset}_groundtruth_{K}.ivecs')
        gt_d_path = os.path.join(path, f'{dataset}_groundtruth_dist_{K}.fvecs')

        # gt_path_bin = os.path.join(path, f'{dataset}_groundtruth.ibin')
        # gt_d_path_bin = os.path.join(path, f'{dataset}_groundtruth_dist.fbin')
        # gt_path_bin = os.path.join(path, f'groundtruth.smallest.img0.1M.ibin')
        # gt_path = os.path.join(path, f'groundtruth.img_0.1M.text_0.2k.ibin')
        
        # gt = read_ibin(gt_path_bin)
        # gt = read_ivecs(gt_path)
        # gt_d = read_fvecs(gt_d_path)
        # print(1)
        # gt = read_ibin(gt_path_bin)
        write_ivecs(gt_path, GT_I)
        write_fvecs(gt_d_path, GT_D)

        
        # gt = read_ibin(gt_path_bin)              
        # print(np.allclose(GT_I, gt))      
        # write_ibin(gt_path_bin, GT_I)
        
        # write_ivecs(gt_path, GT_I)
        # write_fvecs(gt_d_path, GT_D)
