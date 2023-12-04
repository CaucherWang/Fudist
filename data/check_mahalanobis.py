import numpy as np
import pandas as pd
import scipy as stats
from utils import *
from numba import njit

# @njit
def mahalanobis_distance(x, y, cov):
    x = np.asarray(x)
    y = np.asarray(y)
    cov = np.asarray(cov)
    
    # Check if the covariance matrix is valid
    assert cov.shape[0] == cov.shape[1], "Covariance matrix must be square"
    # assert cov.shape[0] == x.shape[0], "Covariance matrix size must match vector size"
    
    # Compute the inverse of the covariance matrix
    inv_cov = np.linalg.inv(cov)
    
    # Compute the difference between x and y
    diff = x - y
    
    # Compute the Mahalanobis distance
    distance = np.dot(np.dot(diff, inv_cov), diff.T)
    
    return distance

import numpy as np

def calculate_covariance_matrix(data):
    data = np.asarray(data)
    
    # Check if data is a 2D array
    assert data.ndim == 2, "Data must be a 2D array"
    
    # Calculate the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    
    return cov_matrix

def mahalanobis_distance_gpt(base_matrix, test_matrix):
    """
    Compute the Mahalanobis distance between a base matrix and a test matrix.

    Parameters:
    base_matrix (np.array): The base matrix.
    test_matrix (np.array): The test matrix.

    Returns:
    distance (float): The Mahalanobis distance.
    """
    # Compute the covariance matrix of the base matrix
    cov_matrix = np.cov(base_matrix.T)

    # Compute the inverse of the covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Compute the mean of the base matrix
    mean_matrix = np.mean(base_matrix, axis=0)

    # Compute the difference between the test matrix and the mean matrix
    diff_matrix = test_matrix - mean_matrix

    # Compute the Mahalanobis distance
    distance = np.sqrt(np.dot(np.dot(diff_matrix, inv_cov_matrix), diff_matrix.T))

    return distance

def find_positions(S, X):
    res = []
    idx = 0
    for s in S:
        if idx % 10000 == 0:
            print(idx)
        idx += 1
        position = np.where(np.all(X == s, axis=1))
        assert(len(position) > 0)
        res.append(position[0])
    return res

# def find_positions(S, X):
#     # Subtract each vector of S from each vector of X
#     diff = X[:, np.newaxis, :] - S

#     # Find the row in X where the resulting vector is zero
#     zero_rows = np.all(np.isclose(diff, 0), axis=2)

#     # Find the index of the first zero row for each vector of S
#     zero_indices = np.argmax(zero_rows, axis=0)

#     # Check if there is no zero row for each vector of S
#     no_match = np.logical_not(np.any(zero_rows, axis=0))

#     # Set the index to -1 for vectors with no match
#     zero_indices[no_match] = -1

#     # Print the result
#     return zero_indices

import os
import random
source = './data/'
dataset = 'webvid1M'
path = os.path.join(source, dataset)
data_path = os.path.join(path, f'{dataset}_base.fbin')
# query_path = os.path.join(path, f'{dataset}_query.fvecs')
# data_path = os.path.join(path, f'{dataset}_sample_query.fbin')
query_path = os.path.join(path, f'{dataset}_query.fbin')
sample_query_path = os.path.join(path, f'{dataset}_sample_query.fbin')
# sample_query_gt_path = os.path.join(path, f'groundtruth.img0.1M.text0.1M.ibin')
# sample_query_smallest_gt_path = os.path.join(path, f'groundtruth.img0.1M.text0.0.1M_smallest.ibin')
# sample_query_largest_gt_path = os.path.join(path, f'groundtruth.img0.1M.text0.0.1M_largest.ibin')

# sample_query_gt = read_ibin(sample_query_gt_path)

# read data vectors
print(f"Reading {dataset} from {data_path}.")
# X = read_fvecs(data_path)
xt = read_fbin(data_path)
if 'laion' in dataset:
    xt = L2_norm_dataset(xt)
print(xt.shape)

# read data vectors
# print(f"Reading {dataset} from {query_path}.")
# # Q = read_fvecs(query_path)
# xq = read_fbin(query_path)
# if 'laion' in dataset:  
#     xq = L2_norm_dataset(xq)
# # norms = np.linalg.norm(Q, axis=1)
# print(xq.shape)

# sq = read_fbin(sample_query_path)
# if 'laion' in dataset:
#     sq = L2_norm_dataset(sq)
# sq_gt = read_ibin(sample_query_gt_path)

# sq1_pos = np.random.choice(sq.shape[0], 100000, replace=False)
# sq1 = sq[sq1_pos]
# sq1_gt = sq_gt[sq1_pos]
# write_fbin(os.path.join(path, f'{dataset}_sample_query_uniform.fbin'), sq1)
# write_ibin(os.path.join(path, f'groundtruth.img0.1M.text0.0.1M_uniform.ibin'), sq1_gt)

# sq_uniform = read_fbin(os.path.join(path, f'{dataset}_sample_query_uniform.fbin'))
# sq1 = read_fbin(os.path.join(path, f'{dataset}_sample_query_smallest.fbin'))
# sq2 = read_fbin(os.path.join(path, f'{dataset}_sample_query_largest.fbin'))

cov_matrix = calculate_covariance_matrix(xt)
print("Covariance matrix:")
# print(cov_matrix)
xt_mean = np.mean(xt, axis=0)
xb_xb_mean_md = []
xq_xb_mean_md = []
sq1_xb_mean_md = []
sq2_xb_mean_md = []
check_first = 2000
# sample check_fist elements from xt and xq
rows = np.random.choice(xt.shape[0], check_first, replace=False)
xt_sample = xt[rows, :]
# rows = np.random.choice(xq.shape[0], check_first, replace=False)
# xq_sample = xq[rows, :]
# rows = np.random.choice(sq1.shape[0], check_first, replace=False)
# sq1_sample = sq1[rows, :]
# rows = np.random.choice(sq2.shape[0], check_first, replace=False)
# sq2_sample = sq2[rows, :]
# rows = np.random.choice(sq_uniform.shape[0], check_first, replace=False)
# sq_uniform_sample = sq_uniform[rows, :]
idx = 0

for x in xt:
    if(idx % 1000 == 0):
        print(idx)
    idx+=1
    distance = mahalanobis_distance(x, xt_mean, cov_matrix)
    xb_xb_mean_md.append(distance)

# xb_xb_mean_md = mahalanobis_distance( xt[:5, :], xt_mean,  cov_matrix)

# idx = 0
# for q in xq_sample:
#     if(idx % 1000 == 0):
#         print(idx)
#     idx+=1
#     distance = mahalanobis_distance(q, xt_mean, cov_matrix)
#     xq_xb_mean_md.append(distance)

idx = 0
for q in sq1_sample:
    if(idx % 1000 == 0):
        print(idx)
    idx+=1
    distance = mahalanobis_distance(q, xt_mean, cov_matrix)
    sq1_xb_mean_md.append(distance)

idx = 0
for q in sq2_sample:
    if(idx % 1000 == 0):
        print(idx)
    idx+=1
    distance = mahalanobis_distance(q, xt_mean, cov_matrix)
    sq2_xb_mean_md.append(distance)

idx = 0
for q in sq_uniform_sample:
    if(idx % 1000 == 0):
        print(idx)
    idx += 1
    distance = mahalanobis_distance(q, xt_mean, cov_matrix)
    xq_xb_mean_md.append(distance)

# xq_xb_mean_md = np.array(xq_xb_mean_md)
# # find the smallest 10% and largest 10% indexes of the Mahalanobis distance
# num_select = int(0.1 * xq_xb_mean_md.shape[0])
# print(num_select)
# largest_indexes = np.argpartition(xq_xb_mean_md, -num_select)[-num_select:]
# smallest_indexes = np.argpartition(xq_xb_mean_md, num_select)[:num_select]
# write_fbin(os.path.join(path, f'{dataset}_sample_query_largest.fbin'), sq[largest_indexes, :])
# write_fbin(os.path.join(path, f'{dataset}_sample_query_smallest.fbin'), sq[smallest_indexes, :])
# sort xq_xb_mean_md and xb_xb_mean_md
# xb_xb_mean_md.sort()
# xq_xb_mean_md.sort()
    
# plot the distribution of the Mahalanobis distance
import matplotlib.pyplot as plt
plt.hist(xb_xb_mean_md, bins=100, alpha=0.5, label='base')
plt.hist(xq_xb_mean_md, bins=100, alpha=0.5, label='query-uniform')
plt.hist(sq1_xb_mean_md, bins=100, alpha=0.5, label='query-furthest')
plt.hist(sq2_xb_mean_md, bins=100, alpha=0.5, label='query-closest')
plt.legend(loc='upper right')
plt.savefig(f'{dataset}_mahalanobis_sq.png')