from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import LocallyLinearEmbedding
import os
import numpy as np
import math
import random
import time
import faiss
import struct
import os
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

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

def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)
                
def to_ivecs(array: np.ndarray, filename: str):
    print(f"Writing File - {filename}")
    topk = (np.int32)(array.shape[1])
    array = array.astype(np.int32)
    topk_array = np.array([topk] * len(array), dtype='int32')
    file_out = np.column_stack((topk_array, array.view('int32')))
    file_out.tofile(filename)

def rate_limited_imap(f, l):
    """A threaded imap that does not produce elements faster than they
    are consumed"""
    pool = ThreadPool(32)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i[0], i[1], ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()
    pool.close()
    pool.join()

def sanitize(x):
    """ convert array to a c-contiguous float array """
    # return np.ascontiguousarray(x.astype('float32'))
    return np.ascontiguousarray(x, dtype='float32')

def dataset_iterator(x, preproc, bs):
    """ iterate over the lines of x in blocks of size bs"""

    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    # def prepare_block((i0, i1)):
    def prepare_block(i0, i1):
        xb = sanitize(x[i0:i1])
        return i0, preproc.apply_py(xb)

    return rate_limited_imap(prepare_block, block_ranges)

class IdentPreproc:
    """a pre-processor is either a faiss.VectorTransform or an IndentPreproc"""

    def __init__(self, d):
        self.d_in = self.d_out = d

    def apply_py(self, x):
        return x

# This is the modified CPU version of compute_GT from Faiss.
# Performs exhaustive search to find ground truth nearest neighbors.
def compute_GT_CPU(xb, xq, gt_sl):
    nq_gt, _ = xq.shape
    print("compute GT CPU")
    t0 = time.time()

    gt_I = np.zeros((nq_gt, gt_sl), dtype='int64')
    gt_D = np.zeros((nq_gt, gt_sl), dtype='float32')
    heaps = faiss.float_maxheap_array_t()
    heaps.k = gt_sl
    heaps.nh = nq_gt
    heaps.val = faiss.swig_ptr(gt_D)
    heaps.ids = faiss.swig_ptr(gt_I)
    heaps.heapify()
    bs = 10 ** 5

    n, d = xb.shape
    xqs = sanitize(xq[:nq_gt])

    db_gt = faiss.IndexFlatL2(d)

    # compute ground-truth by blocks of bs, and add to heaps
    for i0, xsl in dataset_iterator(xb, IdentPreproc(d), bs):
        db_gt.add(xsl)
        D, I = db_gt.search(xqs, gt_sl)
        I += i0
        heaps.addn_with_ids(
            gt_sl, faiss.swig_ptr(D), faiss.swig_ptr(I), gt_sl)
        db_gt.reset()
    heaps.reorder()

    print("GT CPU time: {} s".format(time.time() - t0))
    
    data_ids = []
    data_dis = []
    for i in range(len(gt_I)):
        candidate = []   
        dis_candidate = []
        for j in range(gt_sl):
            candidate.append(gt_I[i][j])
            dis_candidate.append(gt_D[i][j])
        data_ids.append(np.array(candidate))
        data_dis.append(np.array(dis_candidate))
        
    data_ids = np.array(data_ids)
    data_dis = np.array(data_dis)

    
    return data_ids, data_dis

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

source = './data/'
dataset = 'gauss'
for dim in [1000, 2000, 4000]:
    print('dim', dim)
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}{dim}_base.fvecs')

    X = read_fvecs(data_path)
    X = np.unique(X, axis = 0)
    print(X.shape)
    X = X[:100000]
    nnn = 100
    gt_i, gt_d = compute_GT_CPU(X, X, nnn)
    print('gt finished')

    import pandas as pd
    from geomle import mle
    print(mle(pd.DataFrame(X), average = True, k2 = 99, dist = gt_d)[0])

# import skdim
# #generate data : np.array (n_points x n_dim). Here a uniformly sampled 5-ball embedded in 10 dimensions

# #estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point):
# lpca = skdim.id.lPCA().fit_pw(X,
#                               precomputed_knn = gt_i,
#                               n_neighbors = nnn,
#                               n_jobs = 32)
                            
# #get estimated intrinsic dimension
# print(np.mean(lpca.dimension_pw_)) 




# # Define the number of nearest neighbors to use
# n_neighbors = 100

# # Initialize the LLE object
# lle = LocallyLinearEmbedding(n_neighbors=n_neighbors+1, n_components=2)

# # Fit the LLE object to the data
# lle.fit(X)

# # Calculate the locally intrinsic dimensionality using the MLE method
# lid = lle.inverse_transform(lle.transform(X)).sum(axis=1).mean()

# print("The locally intrinsic dimensionality of the dataset is:", lid)

# def l2_distance(a, b):
#     """
#     Compute the L2 distance between two vectors a and b.
#     """
#     return np.sum((a - b)**2)

# def ComputeIntrinsicDimensionality(dataset, SampleQty = 1000000):
#     dist = []
#     DistMean = 0
#     for n in range(SampleQty):
#         r1 = random.randint(0, len(dataset))
#         r2 = random.randint(0, len(dataset))      
#         obj1 = dataset[r1]
#         obj2 = dataset[r2]
#         d = l2_distance(obj1, obj2)
#         dist.append(d)
#         DistMean += d
#     DistMean /= float(SampleQty)
#     DistSigma = 0
#     for i in range(SampleQty):
#         DistSigma += (dist[i] - DistMean) * (dist[i] - DistMean)
#     DistSigma /= float(SampleQty)
#     IntrDim = DistMean * DistMean / (2 * DistSigma)
#     return IntrDim

# print(ComputeIntrinsicDimensionality(X))