import time
import struct
import os
import faiss
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import matplotlib.pyplot as plt
import parallel_sort

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
    # fv = np.fromfile(filename, dtype=np.float32)
    fv = np.memmap(filename, dtype='float32', mode='r+')
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    # if not all(fv.view(np.int32)[:, 0] == dim):
    #     raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def read_ivecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
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

gt_D = gt_D[: , 1:]
gt_I = gt_I
sorted_gt_D = gt_D[gt_D[:, 0].argsort()]
# for index in [10, 100, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 99999]:
#     print(index, sorted_gt_D[index][0])

close_pairs = np.where( (gt_D < 10000))
print(len(close_pairs))
knn_overlap = []
id_map = {}
for i in range(len(close_pairs[0])):
    id1 = close_pairs[0][i]
    num = close_pairs[1][i]
    id2 = gt_I[id1][num]
    if id1 == id2:
        continue
    if id1 not in id_map:
        id_map[id1] = 1
    else:
        id_map[id1]+=1
    if id2 not in id_map:
        id_map[id2] = 1
    else:
        id_map[id2]+=1
    knn1 = gt_I[id1]
    knn2 = gt_I[id2]
    knn2_dist = gt_D[id2]
    knn1_dist = gt_D[id1]

    # if id2 < 100000:
    #     knn2 = gt_I[id2]
    #     knn2_dist = gt_D[id2]
    # else:
    #     knn2, knn2_dist = compute_GT_CPU(X, X[id2:id2+1], 100)
    #     knn2 = knn2[0, 1:]
    #     knn2_dist = knn2_dist[0, 1:]
        
    knn_overlap.append([len(np.intersect1d(knn1[:5], knn2[:5])), len(np.intersect1d(knn1[:10], knn2[:10])), len(np.intersect1d(knn1[:20], knn2[:20])), len(np.intersect1d(knn1[:50], knn2[:50])), len(np.intersect1d(knn1, knn2))])

print(len(knn_overlap))
print(f"#points {len(id_map)}")
id_map = dict(sorted(id_map.items(), key=lambda item: item[1], reverse=True))
knn_overlap = np.array(knn_overlap)
means = np.mean(knn_overlap, axis=0)
percents = np.percentile(knn_overlap, q = [5, 10,25,50,75], axis = 0)
print(means)
print(percents)

# top1 = sorted_gt_D[:, 0]
# k = 99
# top1 = gt_D[:, k-1]
# plt.hist(top1, bins=30, edgecolor='black')
# plt.xlabel(f'Top-{k} Distance')
# plt.ylabel('Frequency')
# plt.title(f'Distribution of top-{k} distance')
# plt.savefig(f'./figures/{dataset}-top{k}-dist.png')
# X_dis = parallel_sort.sort(top1)
# for percent in [0.001, 0.01, 0.1, 0.3, 0.5, 0.75,0.9, 0.99]:
#     print(f'percent: {percent}, value: {X_dis[int(percent * len(X_dis))]}')
