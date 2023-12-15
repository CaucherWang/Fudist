from utils import *
from queue import PriorityQueue
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density # adds projection='scatter_density'


fig = plt.figure(figsize=(6,4.8))

def read_nsw(filename, numVectors):
    data = np.memmap(filename, dtype='uint64', mode='r')
    print(f'edge number * 2 = {len(data) - numVectors}')
    vectors = []
    cur_pos = 0
    for i in range(numVectors):
        vectorDim = int(data[cur_pos])
        vector = data[cur_pos + 1: cur_pos + 1 + vectorDim]
        vectors.append(vector)
        cur_pos += 1 + vectorDim
    return vectors
    # with open(filename, 'rb') as f:
    #     vectors = []
    #     for _ in range(numVectors):
    #         vectorDim = np.frombuffer(f.read(8), dtype=np.uint64)[0]    # size_t = unsigned long
    #         vector = np.frombuffer(f.read(vectorDim * 8), dtype=np.uint64)
    #         vectors.append(vector)

    # return vectors

class node:
    def __init__(self, id, dist):
        self.id = id
        self.dist = dist
    
    def __lt__(self, other):
        # further points are prior
        return self.dist > other.dist
    
    def __str__(self) -> str:
        return f'id: {self.id}, dist: {self.dist}'
    
def select_ep(n):
    return np.random.choice(n, 1)

def search(graph, vec, query, k, efsearch, visited_points=None, glanced_points=None, eps=None, ndc=0, nhops=0):
    resultSet = PriorityQueue(efsearch)     # len < efSearch
    candidateSet = PriorityQueue()  # unlimited lenth
    visited = set()
    if eps is None:
        eps = select_ep(vec.shape[0])
    for ep in eps:
        candidateSet.put(node(ep, -euclidean_distance(vec[ep], query)))
        
    while not candidateSet.empty():
        top = candidateSet.get()
        if not resultSet.empty() and -top.dist > resultSet.queue[0].dist:
            break
        nhops +=1
        if visited_points is not None:
            visited_points[top.id] += 1
        for nei in graph[top.id]:
            if nei in visited:
                continue
            visited.add(nei)
            if glanced_points is not None:
                glanced_points[nei] += 1
            ndc += 1
            dist = euclidean_distance(vec[nei], query)
            if not resultSet.full() or dist < resultSet.queue[0].dist:
                candidateSet.put(node(nei, -dist))
                if resultSet.full():
                    resultSet.get()
                resultSet.put(node(nei, dist))
                
    while resultSet._qsize() > k:
        resultSet.get()
        
    ret = []
    while not resultSet.empty():
        top = resultSet.get()
        ret.append([top.id, top.dist])
    ret.reverse()
    return ret, ndc, nhops
          
def do_expr(X, G, Q, GT, k, n_rknn, lengths, indegree, file = None):
    f = None 
    if file is not None:
        f = open(file, 'a')
        print(f'write to {file}')
    # efss = [500]
    efss = [60,80, 100, 150, 200, 300, 400, 500, 600, 750, 1000, 1500, 2000]
    # special_q = 1983
    # efss = [100, 200, 300, 400, 500]
    # efss = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 4000, 5000, 5200, 5400, 5600, 5800, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000, 12000, 15000, 18000, 20000, 25000, 30000, 35000, 40000]
    # efss = [50000, 60000, 80000, 100000]
    for efs in efss:
        visited_points = np.zeros(X.shape[0], dtype=np.int32)
        glanced_points = np.zeros(X.shape[0], dtype=np.int32)
        if f is None:
            print(f'efsearch: {efs}', end='\t')
        else:
            f.write(f'efsearch: {efs}\n')
        sum_recall = 0
        sum_ndc = 0
        sum_nhops = 0
        idx = 0
        sum_fp_rknn = 0
        sum_tp_rknn = 0
        sum_fn_rknn = 0
        sum_tp_indegree = 0
        sum_fp_indegree = 0
        sum_fn_indegree = 0
        sum_tp_lengths = 0
        sum_fp_lengths = 0
        sum_fn_lengths = 0
        sum_tps = 0
        sum_fps = 0
        sum_fns = 0
        # print()
        for q in Q:
            # if idx != special_q:
            #     idx+=1
            #     continue
            topk, ndc, nhops = search(G, X, q, k,  efs, visited_points, glanced_points)
            gt = GT[idx][:k]
            recall = get_recall(topk, gt)
            
            tps, fps, fns = decompose_topk_to_3sets(np.array(topk), np.array(gt))
            sum_tps += len(tps)
            sum_fps += len(fps)
            sum_fns += len(fns)
            
            # tp_n_rknn, fp_n_rknn, fn_n_rknn = analyze_results_k_occurrence(tps, fps, fns, n_rknn)
            # sum_fp_rknn += fp_n_rknn
            # sum_tp_rknn += tp_n_rknn
            # sum_fn_rknn += fn_n_rknn
            
            # tp_indegree, fp_indegree, fn_indegree = analyze_results_indegree(tps, fps, fns, indegree)
            # sum_tp_indegree += tp_indegree
            # sum_fp_indegree += fp_indegree
            # sum_fn_indegree += fn_indegree
                        
            # tp_lengths, fp_lengths, fn_lengths = analyze_results_norm_length(tps, fps, fns, lengths)
            # sum_tp_lengths += tp_lengths
            # sum_fp_lengths += fp_lengths
            # sum_fn_lengths += fn_lengths
            
            idx += 1
            if idx % 1000 == 0:
                print(idx)
            sum_ndc += ndc
            sum_nhops += nhops
            sum_recall += recall
        # print()
        recall = sum_recall / Q.shape[0] / k
        ndc = sum_ndc / Q.shape[0]
        nhops = sum_nhops / Q.shape[0]
        if f is None:
            print(f'recall: {recall}, ndc: {ndc}, nhops: {nhops}')
            print(f'\ttp_rknn: {sum_tp_rknn / sum_tps}, fp_rknn: {sum_fp_rknn / sum_fps}, fn_rknn: {sum_fn_rknn / sum_fns}, tp-fn: {sum_tp_rknn / sum_tps - sum_fn_rknn / sum_fns}')
            print(f'{sum_tp_rknn / sum_tps:.2f} & {sum_fp_rknn / sum_fps:.2f} & {sum_fn_rknn / sum_fns:.2f} & {sum_tp_rknn / sum_tps - sum_fn_rknn / sum_fns:.2f}')
            print(f'\ttp_indegree: {sum_tp_indegree / sum_tps}, fp_indegree: {sum_fp_indegree / sum_fps}, fn_indegree: {sum_fn_indegree / sum_fns}, tp-fn: {sum_tp_indegree / sum_tps - sum_fn_indegree / sum_fns}')
            print(f'{sum_tp_indegree / sum_tps:.2f} & {sum_fp_indegree / sum_fps:.2f} & {sum_fn_indegree / sum_fns:.2f} & {sum_tp_indegree / sum_tps - sum_fn_indegree / sum_fns:.2f}')
            print(f'\ttp_lengths: {sum_tp_lengths / sum_tps}, fp_lengths: {sum_fp_lengths / sum_fps}, fn_lengths: {sum_fn_lengths / sum_fns}, tp-fn: {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns}')
            print(f'{sum_tp_lengths / sum_tps:.2f} & {sum_fp_lengths / sum_fps:.2f} & {sum_fn_lengths / sum_fns:.2f} & {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns:.2f}')
        else:
            f.write(f'recall: {recall}, ndc: {ndc}, nhops: {nhops}')
            f.write('\n')
            # f.write(f'\ttp_rknn: {sum_tp_rknn / sum_tps}, fp_rknn: {sum_fp_rknn / sum_fps}, fn_rknn: {sum_fn_rknn / sum_fns}, tp-fn: {sum_tp_rknn / sum_tps - sum_fn_rknn / sum_fns}')
            # f.write('\n')
            # # f.write(f'{sum_tp_rknn / sum_tps:.2f} & {sum_fp_rknn / sum_fps:.2f} & {sum_fn_rknn / sum_fns:.2f} & {sum_tp_rknn / sum_tps - sum_fn_rknn / sum_fns:.2f}')
            # f.write(f'\ttp_indegree: {sum_tp_indegree / sum_tps}, fp_indegree: {sum_fp_indegree / sum_fps}, fn_indegree: {sum_fn_indegree / sum_fns}, tp-fn: {sum_tp_indegree / sum_tps - sum_fn_indegree / sum_fns}')
            # f.write('\n')
            # # f.write(f'{sum_tp_indegree / sum_tps:.2f} & {sum_fp_indegree / sum_fps:.2f} & {sum_fn_indegree / sum_fns:.2f} & {sum_tp_indegree / sum_tps - sum_fn_indegree / sum_fns:.2f}')
            # f.write(f'\ttp_lengths: {sum_tp_lengths / sum_tps}, fp_lengths: {sum_fp_lengths / sum_fps}, fn_lengths: {sum_fn_lengths / sum_fns}, tp-fn: {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns}')
            # f.write('\n')
            # f.write(f'{sum_tp_lengths / sum_tps:.2f} & {sum_fp_lengths / sum_fps:.2f} & {sum_fn_lengths / sum_fns:.2f} & {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns:.2f}')
    if f is not None:
        f.close()
        # write_fbin_simple(f'./figures/{dataset}-hnsw-visit-hub-correlation-ef{efs}.fbin', visited_points)
        # write_fbin_simple(f'./figures/{dataset}-hnsw-glance-hub-correlation-ef{efs}.fbin', glanced_points)
 
def get_recall(topk, groundtruth):
    cnt = 0
    ids = [topk[i][0] for i in range(len(topk))]
    ids = set(ids)
    assert len(ids) == len(topk)
    for id in ids:
        if id in groundtruth:
            cnt += 1
    return cnt 

def analyze_results_hub(topk, gt, n_rknn):
    # find the rknn for each point in topk
    topk_ids = np.array(list(topk[:, 0]), dtype='int32')
    fp_ids = np.setdiff1d(topk_ids, gt)
    tp_ids = np.intersect1d(topk_ids, gt)
    fn_ids = np.setdiff1d(gt, topk_ids)
    
    fp_n_rknn = n_rknn[fp_ids]
    tp_n_rknn = n_rknn[tp_ids]
    fn_n_rknn = n_rknn[fn_ids]
    
    avg_fp_n_rknn = np.average(fp_n_rknn) if len(fp_n_rknn) > 0 else 0
    avg_tp_n_rknn = np.average(tp_n_rknn) if len(tp_n_rknn) > 0 else 0
    avg_fn_n_rknn = np.average(fn_n_rknn) if len(fn_n_rknn) > 0 else 0
    
    return avg_fp_n_rknn, avg_tp_n_rknn, avg_fn_n_rknn
    
    
def get_density_scatter(k_occur, lid):
    x = k_occur
    y = lid
    # y = np.array([np.log2(i) if i > 0 else i for i in y])
    # y = np.log(n_rknn)
    print(np.max(x), np.max(y))

    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)

    def using_mpl_scatter_density(figure, x, y):
        ax = figure.add_subplot(1, 1, 1, projection='scatter_density')
        density = ax.scatter_density(x, y, cmap=white_viridis)
        figure.colorbar(density, label='#points per pixel')

    fig = plt.figure(figsize=(8,6))
    using_mpl_scatter_density(fig, x, y)
    # plt.xlabel('k-occurrence')
    plt.xlabel('Mahalanobis distance')
    # plt.xlabel('local intrinsic dimensionality')
    # plt.ylabel('local intrinsic dimensionality')
    # plt.ylabel('1NN Mahalanobis distance')
    # plt.ylabel('1NN distance')
    plt.ylabel('k-occurrence')
    # plt.tight_layout()
    plt.ylim(0, 3500)
    # plt.yscale('log')
    plt.xlim(0, 400)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k_occurs-lid-scatter.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-k_occurs-lid-scatter.png')
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k-occur-1NN-dist.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-k-occur-1NN-dist.png')
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-ma-dist-1NN-dist.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-ma-dist-1NN-dist.png')
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-ma-dist-1NN-ma-dist.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-ma-dist-1NN-ma-dist.png')
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-lid-1NN-dist.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-lid-1NN-dist.png')
    plt.savefig(f'./figures/{dataset}/{dataset}-query-k-occur-ma-dist.png')
    print(f'save to figure ./figures/{dataset}/{dataset}-query-k-occur-ma-dist.png')

def plot_mahalanobis_distance_distribution(X, Q, dataset):
    ma_dist = get_mahalanobis_distance(X, Q)
    # randomly sample 10,000 base vectors
    S = np.random.choice(X.shape[0], 10000, replace=False)
    base_ma_dist = get_mahalanobis_distance(X, X[S])
    
    plt.hist(base_ma_dist, bins=50, edgecolor='black', label='base', color='orange')
    plt.hist(ma_dist, bins=50, edgecolor='black', label='query', color='steelblue')
    plt.xlabel('Mahalanobis distance')
    plt.ylabel('number of points')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.title(f'{dataset} Mahalanobis distance distribution')
    plt.savefig(f'./figures/{dataset}/{dataset}-mahalanobis-distance-distribution.png')
    print(f'save to file ./figures/{dataset}/{dataset}-mahalanobis-distance-distribution.png')

    
source = './data/'
result_source = './results/'
dataset = 'deep'
idx_postfix = '_plain_1th'
efConstruction = 500
Kbuild = 16
M=16
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    # graph_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_K{Kbuild}.nsw.index')
    index_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}.index{idx_postfix}')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs')
    GT_dist_path = os.path.join(source, dataset, f'{dataset}_groundtruth_dist.fvecs')
    KG = 450
    ind_path = os.path.join(source, dataset, f'{dataset}_ind_{KG}.ibin')
    hnsw_ind_path = os.path.join(source, dataset, f'{dataset}_hnsw_ef{efConstruction}_M{M}_ind_{KG}.ibin')
    kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')
    result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_.log{idx_postfix}_unidepth-level')
    ma_dist_path = os.path.join(source, dataset, f'{dataset}_ma_distance.fbin')
    ma_base_dist_path = os.path.join(source, dataset, f'{dataset}_ma_base_distance.fbin')
    # ma_base_dist_path = os.path.join(source, dataset, f'{dataset}_mahalanobis_base_gpu.fbin')
    # ma_1nn_dist_path = os.path.join(source, dataset, f'{dataset}_ma_1nn_distance.fbin')
    
    # recalls = []
    # with open(result_path, 'r') as f:
    #     lines = f.readlines()
    #     for i in range(3, len(lines), 7):
    #         line = lines[i].strip().split(',')[:-1]
    #         recalls.append(np.array([int(x) for x in line]))

    # low_recall_positions = []
    # for recall in recalls:
    #     low_recall_positions.append(np.where(recall < 40)[0])
    X = read_fvecs(base_path)
    # G = read_hnsw_index_aligned(index_path, X.shape[1])
    # G = read_hnsw_index_unaligned(index_path, X.shape[1])
    Q = read_fvecs(query_path)
    # Q = read_fvecs(query_path)
    GT = read_ivecs(GT_path)
    # GT_dist = read_fvecs(GT_dist_path)
    KGraph = read_ivecs(kgraph_path)
    lengths = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'))
    # q_lengths = get_query_length(X, Q)
    n_rknn = read_ibin_simple(ind_path)
    # indegree = get_indegree_list(G)
    # write_ibin_simple(hnsw_ind_path, indegree)
    # indegree = read_ibin_simple(hnsw_ind_path)
    # revG = get_reversed_graph_list(G)
    
    # plot_mahalanobis_distance_distribution(X, Q, dataset)
        
    # q_lids = get_lids(GT_dist[:, :50], 50)
    # gt0_dist = GT_dist[:, 0]
    # get_density_scatter(q_lids, gt0_dist)
    # lid_indexes = np.argsort(q_lids)
    # ma_dist = get_mahalanobis_distance(X, Q)
    # write_fbin_simple(ma_dist_path, ma_dist)
    # ma_dist = read_fbin_simple(ma_dist_path)
    # get_density_scatter(ma_dist, n_rknn)
    
    ma_base_dist = get_mahalanobis_distance(X, X)
    # write_fbin_simple(ma_base_dist_path, ma_base_dist)
    # ma_base_dist = read_fbin_simple(ma_base_dist_path)
    # plt.hist(ma_base_dist, bins=50, edgecolor='black', label='base', color='orange')
    # print(np.min(ma_base_dist), np.max(ma_base_dist))
    # plt.savefig(f'./figures/{dataset}/{dataset}-mahalanobis-base-distance-distribution.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-mahalanobis-base-distance-distribution.png')
    get_density_scatter(ma_base_dist, n_rknn)
    
    # get_density_scatter(ma_dist, GT_dist[:, 0])
    # print(np.corrcoef(ma_dist, GT_dist[:, 0]))
    
    # ma_1nn_dist = get_1nn_mahalanobis_distance(X, Q, GT)
    # write_fbin_simple(ma_1nn_dist_path, ma_1nn_dist)
    # ma_1nn_dist = read_fbin_simple(ma_1nn_dist_path)
    # get_density_scatter(ma_dist, ma_1nn_dist)
    # print(np.corrcoef(ma_dist, ma_1nn_dist))
    
    # query_ma_dist = get_mahalanobis_distance(X, Q)
    # ma_indexes = np.argsort(query_ma_dist)
    # ma_indexes = np.argsort(ma_1nn_dist)
    npart = 10
    
    
    # diff_q_lids = []
    # for i in range(2000):
    #     q_id = k_occurs_indexes[i]
    #     if q_lids[q_id] < 10:
    #         diff_q_lids.append(q_id)
    
    # print(len(diff_q_lids))
    # diff_q_lids = np.array(diff_q_lids)
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_k_occur_harder_than_lid'), GT[diff_q_lids])
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_k_occur_harder_than_lid'), Q[diff_q_lids])
    
    # diff_q_k_occurs = []
    # for i in range(2000):
    #     q_id = lid_indexes[-i-1]
    #     if q_k_occurs[q_id] > 70:
    #         diff_q_k_occurs.append(q_id)
    
    # print(len(diff_q_k_occurs))
    # diff_q_k_occurs = np.array(diff_q_k_occurs)
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_lid_harder_than_k_occur'), GT[diff_q_k_occurs])
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_lid_harder_than_k_occur'), Q[diff_q_k_occurs])
 
        
    
    # lids4k_occur_mins = []
    # for i in range(500):
    #     q_id = k_occurs_indexes[i]
    #     lids4k_occur_mins.append(q_lids[q_id])
    #     print(q_k_occurs[q_id], end = ',')
        
    # # lids4k_occur_mins = np.array(lids4k_occur_mins)
    # plt.hist(q_lids, bins=50, edgecolor='black')
    # plt.xlabel('local intrinsic dimensionality')
    # plt.ylabel('number of queries')
    # # plt.title(f'{dataset} lid distrib. on \n 500 queries of min k_occurs')
    # plt.title(f'{dataset} lid distrib.')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-lid-distribution.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-query-lid-distribution.png')
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-lid-distribution-min-k_occurs.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-query-lid-distribution-min-k_occurs.png')
    
    
    # plot the scatter point figure of these two
    # get_density_scatter(q_k_occurs, GT_dist[:, 0])
    
    # for i in range(500):
    #     q_id = k_occurs_indexes[-i - 1]
    #     gt_dist_min = GT_dist[q_id][0]
    #     gt_dist_max = GT_dist[q_id][-1]
    #     print(f'{i}: q_id: {q_id},  k_occur:{q_k_occurs[q_id]}, lid: {q_lids[q_id]}, gt_dist_min: {gt_dist_min}, gt_dist_max: {gt_dist_max}, gt_dist_range: {gt_dist_max - gt_dist_min}')
    
    # for i in range(500):
    #     q_id = lid_indexes[-i - 1]
    #     gt_dist_min = GT_dist[q_id][0]
    #     gt_dist_max = GT_dist[q_id][-1]
    #     print(f'{i}: q_id: {q_id},  lid: {q_lids[q_id]},  k_occur:{q_k_occurs[q_id]}, gt_dist_min: {gt_dist_min}, gt_dist_max: {gt_dist_max}, gt_dist_range: {gt_dist_max - gt_dist_min}')


    # plt.hist(q_k_occurs, bins=100)
    # plt.xlabel('k_occurs')
    # plt.ylabel('number of queries')
    # plt.title(f'{dataset} k_occurs distribution')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k_occurs-distribution.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-query-k_occurs-distribution.png')
    
    # query_splits = []
    # q_index_splits = []
    # range_max = 300
    # r = range_max - ma_dist.min()
    # part_size = r / npart
    # start = 0
    # for t in range(npart):
    #     part_max = ma_dist.min() + (t+1) * part_size
    #     if t == npart - 1:
    #         part_max = ma_dist.max() + 1
    #     print(f'from {ma_dist.min() + t * part_size} to {part_max}')
    #     pre_start = start
    #     while start < Q.shape[0] and ma_dist[ma_indexes[start]] < part_max:
    #         start += 1
    #     qindexes = ma_indexes[pre_start: start]
    #     q_index_splits.append(qindexes)
    #     query_splits.append(Q[qindexes])
    #     qpath = os.path.join(source, dataset, f'{dataset}_query.fvecs_ma_unidepth-level{t}')
    #     gtpath = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_ma_unidepth-level{t}')
    #     write_fvecs(qpath, Q[qindexes])
    #     write_ivecs(gtpath, GT[qindexes])

    
    # part_size = Q.shape[0] // npart
    # query_splits = []
    # for i in range(npart):
    #     qindexes = ma_indexes[i*part_size:(i+1)*part_size]
    #     # print(f'from {query_ma_dist[qindexes[0]]} to {query_ma_dist[qindexes[-1]]}')
    #     query_splits.append(Q[qindexes])
    #     qpath = os.path.join(source, dataset, f'{dataset}_query.fvecs_ma_level{i}')
    #     gtpath = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_ma_level{i}')
    #     write_fvecs(qpath, Q[qindexes])
    #     write_ivecs(gtpath, GT[qindexes])
    
    # find the indexes of 200 maximum k_occurs
    # max_q_k_occurs_indexes = np.argsort(q_k_occurs)[::-1][:200]
    
    
    # hard_q_lengths = q_lengths[min_q_k_occurs_indexes]
    # easy_q_lengths = q_lengths[max_q_k_occurs_indexes]
    # print(np.average(hard_q_lengths), np.average(easy_q_lengths))
    
    # hard_queries = Q[min_q_k_occurs_indexes]
    # hard_ground_truth = GT[min_q_k_occurs_indexes]
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_hard'), hard_ground_truth)
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_hard'), hard_queries)
    # easy_queries = Q[max_q_k_occurs_indexes]
    # easy_ground_truth = GT[max_q_k_occurs_indexes]
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_easy'), easy_queries)
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_easy'), easy_ground_truth)
    
        
        

    # print(count_edges_list(G))
    # print(get_graph_quality_list(G, KGraph, KG))
    # get_indegree_hist(indegree, dataset, 'hnsw')
    
    # find the elements with the largest indegree
    # get_outlink_density(indegree, G, X.shape[0])
            
    # insertion_order = np.arange(X.shape[0])[::-1]
    # print(np.corrcoef(insertion_order, indegree))
    # print(np.corrcoef(n_rknn, indegree))
    
    # print(get_skewness(indegree))
    # print(np.corrcoef(np.arange(1000000), indegree))
    # print(np.corrcoef(n_rknn, indegree))
    # # plot the scatter plot of n_rknn and indegree
    # import matplotlib.pyplot as plt
    # plt.plot(n_rknn, indegree, 'o', color='black', markersize=0.5)
    # plt.xlabel('indegree of kNN graph')
    # plt.ylabel('indegree of NSW')
    # plt.xlim(0, 1200)
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title(f'{dataset} indegree correlation')
    # plt.savefig(f'./figures/{dataset}-hnsw-indegree-correlation.png')

    
        
    # k = 50
    # for i in range(npart):
    #     do_expr(X, G, query_splits[i], GT[q_index_splits[i]], k, n_rknn, lengths, indegree, file = result_path + str(i))
    # write_ibin_simple(f'./figures/{dataset}-Hnsw-visit-hub-correlation.ibin', visited_points)
    # write_ibin_simple(f'./figures/{dataset}-Hnsw-glance-hub-correlation.ibin', glanced_points)
    
    # # print(np.corrcoef(n_rknn, glanced_points))
    # plt.close()
    # plt.plot(indegree, glanced_points, 'o', color='black', markersize=0.1)
    # plt.xlabel('Indegree of points')
    # plt.ylabel('Visted times')
    # # plt.xlim(0, 1200)
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.title(f'{dataset} indegree correlation')
    # plt.savefig(f'./figures/{dataset}-Hnsw-glance-hub-correlation.png')
    
    # filter the points with glanced_points > 200
    # filter_pos = np.where(glanced_points > 200)[0]
    # filter_r_knn = n_rknn[filter_pos]
    # print(np.min(filter_r_knn), np.percentile(filter_r_knn, 1), np.percentile(filter_r_knn, 5), np.percentile(filter_r_knn, 50))

    