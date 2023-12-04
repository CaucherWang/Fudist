from utils import *
import os
from pprint import pprint

source = './data/'
datasets = ['glove1.2m']
ef = 500
M = 16

dataset = datasets[0]
path = os.path.join(source, dataset)
data_path = os.path.join(path, f'{dataset}_base.fvecs')
query_path = os.path.join(path, f'{dataset}_query.fvecs')
gt_path = os.path.join(path, f'{dataset}_groundtruth.ivecs')
gt_dist_path = os.path.join(path, f'{dataset}_groundtruth_dist.fvecs')
neighbors_path = os.path.join(path, f'{dataset}_ef{ef}_M{M}.index_plain_neighbors')
knn_path = os.path.join(path, f'{dataset}_self_groundtruth.ivecs')
knn_dist_path = os.path.join(path, f'{dataset}_self_groundtruth_dist.fvecs')

X = read_fvecs(data_path)
Q = read_fvecs(query_path)
GT = read_ivecs(gt_path)
GT_DIST = read_fvecs(gt_dist_path)
KNN = read_ivecs(knn_path)
KNN_DIST = read_fvecs(knn_dist_path)
neighbors = read_ivecs(neighbors_path)

q_num = 40
query = Q[q_num]
gt = GT[q_num]
gt_dist = GT_DIST[q_num]
neighbor = neighbors[gt]
print(1)
