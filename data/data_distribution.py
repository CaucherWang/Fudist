from sklearn.cluster import dbscan
import os
import numpy as np
from utils import *
from estimators import *

source = './data/'
datasets = ['glove1.2m']

dataset = datasets[0]
path = os.path.join(source, dataset)
data_path = os.path.join(path, f'{dataset}_base.fvecs')
ind_path = os.path.join(path, f'{dataset}_ind.ibin')
ind_path1 = os.path.join(path, f'{dataset}_ind.ibin_shuf1')
ind_path2 = os.path.join(path, f'{dataset}_ind.ibin_shuf2')
gt_path = os.path.join(path, f'{dataset}_self_groundtruth_dist.fvecs')
pos_path2 = os.path.join(path, f'{dataset}_pos2.ibin')
pos_path1 = os.path.join(path, f'{dataset}_pos1.ibin')
print(f"Reading positions from {pos_path2}")
pos2 = ibin_read_simple(pos_path2)  # pos[new_id] = old_id
pos1 = ibin_read_simple(pos_path1)  # pos[new_id] = old_id

X = read_fvecs(data_path)
GT_dist = read_fvecs(gt_path)
inds = np.array(ibin_read_simple(ind_path))
inds1 = np.array(ibin_read_simple(ind_path1))
inds2 = np.array(ibin_read_simple(ind_path2))

sep = np.where(inds < 2)[0]
sep1 = np.where(inds1 < 2)[0]
sep2 = np.where(inds2 < 2)[0]
len0 = sep.shape[0]
len1 = sep1.shape[0]
len2 = sep2.shape[0]
lenX = X.shape[0]
print(sep.shape, sep1.shape, sep2.shape)
print(X.shape)
print('expectation')
print(len1 * len2 / (lenX), len1 * len2 / (lenX) / min(len1, len2))
print(len0 * len1 / (lenX), len0 * len1 / (lenX) / min(len0, len1))
print(len0 * len2 / (lenX), len0 * len2 / (lenX) / min(len0, len2))

sep1 = np.array([pos1[id] for id in sep1])
sep2 = np.array([pos2[id] for id in sep2])


print(len(np.intersect1d(sep1, sep2)), len(np.intersect1d(sep1, sep2)) / min(sep1.shape[0], sep2.shape[0]))
print(len(np.intersect1d(sep1, sep)), len(np.intersect1d(sep1, sep)) / min(sep1.shape[0], sep.shape[0]))
print(len(np.intersect1d(sep, sep2)), len(np.intersect1d(sep2, sep)) / min(sep2.shape[0], sep.shape[0]))
# lids = get_lids(GT_dist, 99)
# top_100_dist  = GT_dist[:, 99]
# print("pearson correlation for lids and indegrees")
# print(np.corrcoef(lids, inds))
# print('pearson correlation for top-100 dist. and indegrees')
# print(np.corrcoef(top_100_dist, inds))


# sample for lid
# separate_points = np.array([28703, 92178, 117275, 354498, 295839])
# found_points = np.array([35054, 887960, 968312, 138404, 1124581, 968339])
# GT_seperate_points = GT[separate_points]
# GT_found_points = GT[found_points]
# sep_lids = get_lids(GT_seperate_points, 99)
# found_lids = get_lids(GT_found_points, 99)
# print(np.mean(sep_lids))
# print(np.mean(found_lids))

# l2_norm = calculation_L2_norm(X)
# print(len(l2_norm))