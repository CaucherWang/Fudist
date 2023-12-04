import os
import numpy as np
import struct
import time
from utils import *


source = './data/'
# datasets = ['deep', 'gist', 'glove1.2m', 'msong', 'sift', 'tiny5m', 'ukbench', 'word2vec']
datasets = ['deep']

                
if __name__ == "__main__":
    
    for dataset in datasets:
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs_shuf4')
        data_path_bin = os.path.join(path, f'{dataset}_base.fbin')
        learn_path = os.path.join(path, f'{dataset}_learn.fbin')
        sample_query_path_bin = os.path.join(path, f'{dataset}_sample_query.fbin')
        sample_query_path = os.path.join(path, f'{dataset}_sample_query.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        query_path_bin = os.path.join(path, f'{dataset}_query.fbin')
        gt_path_bin = os.path.join(path, f'{dataset}_groundtruth.ibin')
        gt_path = os.path.join(path, f'{dataset}_groundtruth.ivecs_shuf4')
        gt_dist_path = os.path.join(path, f'{dataset}_groundtruth_dist.fvecs')
        gt_dist_path_bin = os.path.join(path, f'{dataset}_groundtruth_dist.fbin')
        visual_path = os.path.join(path, f'{dataset}_visual.fbin')
        id_query_path = os.path.join(path, f'{dataset}_id_query.fbin')
        gt_id_query_path = os.path.join(path, f'{dataset}_gt_id_query.ibin')
        data_path_norm_bin = os.path.join(path, f'{dataset}_base.fbin_norm')
        data_path_norm = os.path.join(path, f'{dataset}_base.fvecs_norm')
        query_path_norm = os.path.join(path, f'{dataset}_query.fvecs_norm')
        query_path_norm_bin = os.path.join(path, f'{dataset}_query.fbin_norm')

        # read data vectors
        # print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        Q = read_fvecs(query_path)
        GT = read_ivecs(gt_path)
        GT_dist = read_fvecs(gt_dist_path)
        print(1)
        # gt_dis = pair_matrix_innerproduct(Q, GT, X)
        # X = read_fbin(data_path_bin)
        # X = read_fbin_cnt(data_path_bin, 100000000)
        # Q = read_fbin(query_path_bin)
        # T = read_fbin(learn_path)
        # SQ = read_fbin(sample_query_path_bin)
        # GT = read_ibin(gt_path_bin)
        # GT_dist = read_fbin(gt_dist_path_bin)
        # print(GT_dist.shape)
        # V = np.memmap(visual_path, dtype='float32', mode='r')
        # X = read_fvecs(data_path)
        # X = fbin_read_cnt(data_path, 10000000)
        # X = i8bin_read_cnt(data_path, 10000000)
        # D = X.shape[1]
        # print(f"{dataset} of dimensionality {D} of cardinality {X.shape[0]}.")
        
        # X = L2_norm_dataset(X)
        # Q = L2_norm_dataset(Q)
        
        # V = np.concatenate((X, SQ), axis=0)
        # write_fbin(query_path_norm_bin, Q)
        # write_fvecs(query_path_norm, Q)
        # write_fbin(data_path_norm_bin, X)
        # write_fvecs(data_path_norm, X)
        # write_fbin(id_query_path, X[-1000:])
        # write_fbin(visual_path, V)
        # write_fbin(data_path_bin, X)
        # write_fvecs(data_path, X)
        # write_fvecs(query_path, X)
        # write_fvecs(query_path, Q)
        write_ivecs(gt_path, GT)
        # write_fvecs(gt_dist_path, gt_dis)
        # write_fvecs(sample_query_path, SQ)
        # write_fbin(sample_query_path_bin, SQ[:-10000])
        # write_fvecs(query_path, SQ[-10000:])
        # write_fbin(query_path_bin, SQ[-10000:])
        # gt_I, gt_D = compute_GT_CPU(X, X[-1000:], 100)
        # write_ibin(gt_id_query_path, gt_I)
        
        # out_path = os.path.join(path, 'spacev100m_query.fvecs')
        # out_path = './data/spacev1m/spacev10m_base.fvecs'
        # to_fvecs2(out_path, X)