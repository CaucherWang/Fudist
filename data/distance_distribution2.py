import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
import numba
from multiprocessing import Pool, sharedctypes
import parallel_sort

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
    
def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
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
    
def rand_select(nq, nx):
    # randomly select nq numbers from [0,nx)
    return np.random.choice(nx, nq, replace=False)

def ibin_write(x, path: str):
    x = x.astype('int32')
    f = open(path, "wb")
    x.tofile(f)
    
@numba.njit
def compute_ed(matrix1, matrix2):
    # diff = matrix1[:, np.newaxis] - matrix2
    # squared_diff = diff ** 2
    # return np.sum(squared_diff, axis=2)
    distance_matrix = np.zeros((matrix1.shape[0], matrix2.shape[0]))
    for i in range(matrix1.shape[0]):
        for j in range(matrix2.shape[0]):
            distance_matrix[i, j] = np.sum((matrix1[i] - matrix2[j]) ** 2)
    return distance_matrix


# source = '/home/hadoop/wzy/dataset/'
# datasets = ['yandex-text2image1M']
source  = './data/'
datasets = ['sift']
if __name__ == '__main__':
    for dataset in datasets:
        dir = os.path.join(source, dataset)
        data_file = os.path.join(dir, f'{dataset}_base.fvecs')
        query_file = os.path.join(dir, f'{dataset}_query.fvecs')
        # X = fbin_read(data_file)
        X = read_fvecs(data_file)
        X_sampl1 = rand_select(20000, X.shape[0])
        X_sampl2 = rand_select(20000, X.shape[0])
        
        X1 = X[X_sampl1]
        X2 = X[X_sampl2]
        
        # X_dis = X1 @ X2.T
        # XQ_dist = X1 @ Q1.T
        
        # X_dis = compute_ed(X1, X2)
        # X_dis = compute_ed(X1, X2)
        # X_dis = cdist(X1, X2, metric='sqeuclidean')
        X_dis = pairwise_distances(X1, X2, metric='sqeuclidean', n_jobs = -1)
        # XQ_dist = cdist(X1, Q1, metric='sqeuclidean')
        print(X_dis.shape)
        
        
        
        # flatten X_dis, Q_dis, XQ_dist
        X_dis = X_dis.flatten()
        # XQ_dist = XQ_dist.flatten()
        print(X_dis.shape)
        
        #sort X_dis
        X_dis = parallel_sort.sort(X_dis)
        for percent in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.99]:
            print(f'percent: {percent}, value: {X_dis[int(percent * len(X_dis))]}')
        
        
        plt.figure(figsize=(6, 6))
        # Plot the histogram
        plt.hist(X_dis, bins=30, edgecolor='black')  # Adjust the number of bins as needed

        # Add labels and title
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Base-Base')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plot
        plt.savefig(f'./figures/{dataset}_dist.png')