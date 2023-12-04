'''
generate ground truth nearest neighbors
'''

import os
import numpy as np
from utils import *
source = './data/'
datasets = ['deep']

def read_entry_points(filename):
    eps = []
    with open(filename) as f:
        line = f.readline()
        line = line.split(',')
        eps = [int(x) for x in line]
    print(len(eps))
    return eps

if __name__ == '__main__':
    for dataset in datasets:
        # for dim in [50, 100, 150, 200, 250, 300, 500, 750, 1000, 2000, 4000]:
            # path
        path = os.path.join(source, dataset)
        base_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        gt_path = os.path.join(path, f'{dataset}_groundtruth.ivecs')
        index_path = os.path.join(path, f'{dataset}_ef500_M16.index')
        kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')

        X = read_fvecs(base_path)
        # G = read_hnsw_index_aligned(index_path, X.shape[1])
        Q = read_fvecs(query_path)
        # Q = read_fvecs(query_path)
        GT = read_ivecs(gt_path)
        KG = read_ivecs(kgraph_path)
        
        K = 32
        
        eps = read_entry_points(os.path.join(path, f'{dataset}_entry_points.txt'))
        total = len(eps)
        
        for kk in [1,5,10,20,30,50]:
            hit = 0
            for i in range(len(eps)):
                if eps[i] in GT[i][:kk]:
                    hit += 1
            print(hit / total)
        
        # dists = []
        # for i in range(len(eps)):
        #     dists.append(euclidean_distance(X[eps[i]], Q[i]))
            
        # dists = np.array(dists)
        # # percentiles of dists
        # print(np.percentile(dists, [0, 25, 50, 75, 100]))
        # print(np.mean(dists))
        
        
        baseline_dists = []
        rand_ep =  np.random.choice(X.shape[0], 1)
        
        for i in range(len(eps)):
            baseline_dists.append(euclidean_distance(X[rand_ep], Q[i]))
            
        baseline_dists = np.array(baseline_dists)
        # percentiles of dists
        print(np.percentile(baseline_dists, [0, 25, 50, 75, 100]))
        print(np.mean(baseline_dists))
        
        kg_dists = []
        for i in range(X.shape[0]):
            kg_dists.append(euclidean_distance(X[i], X[KG[i][K+1]]))
            
        kg_dists = np.array(kg_dists)
        # percentiles of dists
        print(np.percentile(kg_dists, [0, 25, 50, 75, 100]))
        print(np.mean(kg_dists))
        
        
        
        
