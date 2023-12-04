import time
import struct
import os
import faiss
import numpy as np
import matplotlib.pyplot as plt
import parallel_sort
from numba import njit
from mpl_toolkits.mplot3d import Axes3D
from estimators import *
from utils import *



source = './data/'
datasets = ['glove1.2m']

dataset = datasets[0]
path = os.path.join(source, dataset)
data_path = os.path.join(path, f'{dataset}_base.fvecs')
# gt_path = os.path.join(path, f'{dataset}_groundtruth_dist.fvecs')
gt_path2_id = os.path.join(path, f'{dataset}_self_groundtruth.ivecs')
gt_path2 = os.path.join(path, f'{dataset}_self_groundtruth_dist.fvecs')
# gt_query_D = read_fvecs(gt_path)
# sorted_gt_query_D = gt_query_D[gt_query_D[:, 0].argsort()]
# for index in [10, 100, 500, 1000, 2000, 3000, 5000, 9999]:
#     print(index, sorted_gt_query_D[index][0])
gt_D = read_fvecs(gt_path2)
gt_I = read_ivecs(gt_path2_id)
X = read_fvecs(data_path)
ind_path = os.path.join(path, f'{dataset}_ind.ibin')
print(f"Reading indegree from {ind_path}")
indegree = np.array(ibin_read_simple(ind_path))


top_10_dsitance = gt_D[: , 10]
gt_I = gt_I
# sorted_gt_D = gt_D[gt_D[:, 0].argsort()]
# for index in [10, 100, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 99999]:
#     print(index, sorted_gt_D[index][0])
indegree_gt = get_reversed_knn_number(gt_I)
lids = get_lids(gt_D, 99)
# eps = 20
# min_samples = 100
# cluster_ids = ibin_read(os.path.join(path, f'{dataset}_dbscan_cluster_{eps}_{min_samples}.ibin'))
# concatenate gt_D, indegree, indegree_gt
cell = np.vstack((top_10_dsitance , indegree , indegree_gt, lids))
cell = cell.T
# sort_index = np.argsort(cell[:, 0])
# sorted_arr = cell[sort_index]

# index = np.where(top_10_dsitance >= 100)
# cell = cell[index][:, 0]
# print(cell.shape)
# plt.hist(cell, bins=100, edgecolor='black')
# plt.savefig(f'../figures/{dataset}-0-indegree-top-10-dist-distrib.png')

# dist_thre = 50
# index = np.where(top_10_dsitance >= dist_thre)
# cell = cell[index][:, 1]
# print(cell.shape)
# plt.hist(cell, bins=400, edgecolor='black')
# # plt.xlim(0, 100)
# plt.xlabel(f'Indegree of HNSW points')
# plt.ylabel('Frequency')
# plt.title(f'{dataset} top-10distance>={dist_thre} cardinality {cell.shape[0]}')
# plt.savefig(f'../figures/{dataset}-far-top-10-indegree-distrib.png')

dist_thre = 10
index1 = np.where(top_10_dsitance < dist_thre)
index2 = np.where(indegree < 1)
index = np.intersect1d(index1, index2)
cell = cell[index]
print(cell.shape)
plt.hist(cell, bins=400, edgecolor='black')
# plt.xlim(0, 100)
plt.xlabel(f'Indegree of HNSW points')
plt.ylabel('Frequency')
plt.title(f'{dataset} top-10distance>={dist_thre} cardinality {cell.shape[0]}')
plt.savefig(f'../figures/{dataset}-far-top-10-indegree-distrib.png')




# for i in range(X.shape[0]):
#     if cell[i][2] < 1 and cell[i][0] < 20:
#         knns = gt_D[i]

# cell.sort(axis=2)
# print(1)

# plot scatter figures for top10 distance and indegree





# fig = plt.figure(figsize = (10, 7)) 
# ax = plt.axes(projection ="3d")  
# # plot 3D points figure for cell[0], cell[1] cell[2]
# # plt.scatter(cell[0], cell[1], cell[2], color = "red")
# # Creating a plot using the random datasets   
# ax.scatter3D(cell[0], cell[1], cell[2], color = "red", marker='o', s=1)  
# # ax.set_yscale('log')
# plt.title(f'{dataset} indegree correlation')
# ax.set_xlabel(f'HNSW points\' Indegree')
# ax.set_ylabel(f'KNN graph points\' Indegree')
# ax.set_zlabel(f'top-10 Distance')
# plt.savefig(f'./figures/{dataset}-correlation.png')
#compute pearson coefficient
# print(np.corrcoef(indegree, indegree_gt))
#plot the curve of indegree and indegree_gt
# plt.plot(indegree, indegree_gt, 'o', color='black', markersize=1)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.title(f'{dataset} indegree correlation')
# plt.savefig(f'./figures/{dataset}-indegree-correlation.png')

# plt.plot( indegree_gt,indegree, 'o', color='black', markersize=1)
# plt.xlabel('indegree of knn graph')
# plt.ylabel('hnsw indegree')
# plt.xlim(0, 10)
# plt.ylim(0, 100)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.title(f'{dataset}')
# plt.savefig(f'../figures/{dataset}-indegree-kg-indegree-correlation.png')


# plt.plot( indegree_gt,top_10_dsitance, 'o', color='black', markersize=0.1)
# plt.ylabel('top-10 distance')
# plt.xlabel('indegree of knn graph')
# plt.xscale('log')
# # plt.yscale('log')
# plt.title(f'{dataset} top-10-distance vs indegree correlation')
# plt.savefig(f'../figures/{dataset}-indegree-top10-dist-correlation.png')


# plt.plot(indegree, lids, 'o', color='black', markersize=1)
# plt.xlabel('indegree of HNSW points')
# plt.ylabel('LID of HNSW points')
# plt.xscale('log')
# # plt.yscale('log')
# plt.title(f'{dataset} lid vs indegree correlation')
# plt.savefig(f'./figures/{dataset}-lid-indegree-correlation.png')

# plt.scatter(np.log(indegree), top_10_dsitance, c='black', s=1)
# plt.xlabel('indegree of HNSW points, log scale')
# plt.ylabel('top-10-distance')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.title(f'{dataset} lid vs top-10-distance correlation')
# plt.savefig(f'./figures/{dataset}-lid-top10-distance-correlation.png')



# indegree_gt.sort()

# print("in-degree:")
# mean = np.mean(indegree_gt)
# percents = np.percentile(indegree_gt, q=[5,10,25,50,75,90])
# print(mean)
# print(percents)


# # top1 = sorted_gt_D[:, 0]
# # k = 99
# # top1 = gt_D[:, k-1]
# plt.hist(indegree_gt, bins=100, edgecolor='black')
# plt.xlabel(f'GT indegree')
# plt.ylabel('Frequency')
# plt.yscale('log')
# plt.title(f'Distribution of GT indegrees')
# plt.savefig(f'./figures/{dataset}-gt-indegree-dist.png')
