'''
generate ground truth nearest neighbors
'''

import time
import faiss
import struct
import os
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

source = './data/'
datasets = ['gist']



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


if __name__ == '__main__':
    for dataset in datasets:
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')

        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        D = X.shape[1]
        print(X.shape)
        
        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        Q = read_fvecs(query_path)
        QD = Q.shape[1]
        print(Q.shape)
        
        K = 100
        
        GT_I, GT_D = compute_GT_CPU(X, Q, K)
        print(GT_I.shape)
        
        gt_path = os.path.join(path, f'{dataset}_groundtruth.ivecs')
        
        to_ivecs(GT_I, gt_path)
