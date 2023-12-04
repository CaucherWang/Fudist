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
    # x = np.log2(x)
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
    # plt.xlabel('kGraph')
    # plt.xlabel('Mahalanobis distance')
    # plt.ylabel('k-occurrence')
    # plt.xticks([5000,10000,15000,20000,25000,30000,35000,40000, 45000, 50000, 55000, 60000],['5k','10k', '15k','20k', '25k', '30k', '35k', '40k', '45k', '50k', '55k', '60k'],fontsize=14)
    # plt.xticks([5000,10000,15000,20000,25000,30000,35000,40000, 45000, 50000, 55000, 60000],['5k','10k', '15k','20k', '25k', '30k', '35k', '40k', '45k', '50k', '55k', '60k'],fontsize=14)
    # plt.yticks([5000,10000,15000],['5k','10k', '15k'],fontsize=14)

    # plt.yticks(fontsize=14)
    # plt.ylabel('k-occurrence (32)')
    # plt.ylabel('local intrinsic dimensionality')
    # plt.ylabel('1NN distance')
    # plt.tight_layout()
    # plt.xlim(0, 400)
    # plt.ylim(0, 4000)
    
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k_occurs-lid-scatter.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-k_occurs-lid-scatter.png')
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k-occur-1NN-dist.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-k-occur-1NN-dist.png')
    plt.savefig(f'./figures/{dataset}/{dataset}-query-difficulty.png')
    print(f'save to figure ./figures/{dataset}/{dataset}-query-difficulty.png')


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

    
def resolve_performance_variance_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        ndcs = []
        for line in lines:
            if(len(line) < 30):
                continue
            line = line.strip()
            ndcs = line[:-1].split(',')
            ndcs = [int(x) for x in ndcs]
        return ndcs
    
def resolve_delta_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        deltas = []
        for line in lines:
            if(len(line) < 30):
                continue
            line = line.strip()
            deltas = line[:-1].split(',')
            deltas = [float(x) for x in deltas]
        return deltas

    
source = './data/'
result_source = './results/'
dataset = 'deep'
idx_postfix = '_plain_1th'
efConstruction = 500
Kbuild = 100
M=16
L = 50
R = 32
C = 500
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    # graph_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_K{Kbuild}.nsw.index')
    index_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}.index{idx_postfix}')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs')
    GT_dist_path = os.path.join(source, dataset, f'{dataset}_groundtruth_dist.fvecs')
    KG = 499
    ind_path = os.path.join(source, dataset, f'{dataset}_ind_{KG}.ibin')
    ind_path2 = os.path.join(source, dataset, f'{dataset}_ind_32.ibin')
    hnsw_ind_path = os.path.join(source, dataset, f'{dataset}_hnsw_ef{efConstruction}_M{M}_ind_{KG}.ibin')
    kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')
    result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_.log{idx_postfix}_unidepth-level')
    delta_result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph{Kbuild}_delta.log')
    ma_dist_path = os.path.join(source, dataset, f'{dataset}_ma_distance.fbin')
    ma_base_dist_path = os.path.join(source, dataset, f'{dataset}_ma_base_distance.fbin')
    query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance.log_plain')
    kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log')
    # kgraph_query_performance_log_path2 = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K100_perform_variance.log')
    in_ds_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance.log_in-dataset_plain_1th')
    nsg_query_performance_log_path = os.path.join(result_source, dataset, f'{dataset}_nsg_L{L}_R{R}_C{C}_perform_variance.log')
    query_performance_log_paths = []
    nsg_query_performance_log_paths = []
    for i in range(3, 12):
        query_performance_log_paths.append(os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance.log_plain_shuf{i}'))
    for i in range(3, 8):
        nsg_query_performance_log_paths.append(os.path.join(result_source, dataset, f'{dataset}_nsg_L{L}_R{R}_C{C}_perform_variance.log_shuf{i}'))


    query_performance = np.array(resolve_performance_variance_log(query_performance_log_path))
    query_performances = [query_performance]
    for i in range(9):
        query_performances.append(np.array(resolve_performance_variance_log(query_performance_log_paths[i])))
    query_performances = np.array(query_performances)
    query_performance_avg = np.sum(query_performances, axis=0) / len(query_performances)
    
    nsg_query_performance = resolve_performance_variance_log(nsg_query_performance_log_path)
    nsg_query_performances = [np.array(nsg_query_performance)]
    for i in range(5):
        nsg_query_performances.append(np.array(resolve_performance_variance_log(nsg_query_performance_log_paths[i])))
    nsg_query_performance_avg = np.sum(nsg_query_performances, axis=0) / len(nsg_query_performances)
    kgraph_query_performance = resolve_performance_variance_log(kgraph_query_performance_log_path)
    in_ds_query_performance = resolve_performance_variance_log(in_ds_query_performance_log_path)
    # kgraph_query_performance2 = resolve_performance_variance_log(kgraph_query_performance_log_path2)
    
    n_rknn = read_ibin_simple(ind_path)
    ma_base_dist = read_fbin_simple(ma_base_dist_path)
    
    kgraph_query_performance_hardest = resolve_performance_variance_log(kgraph_query_performance_log_path + '_hardest')
    kgraph_query_performance_hardest2 = resolve_performance_variance_log(kgraph_query_performance_log_path + '2_hardest')
    
    # for i in [4192, 4023]:
    #     # print(f'{i}: {kgraph_query_performance_hardest[i]}, {kgraph_query_performance_hardest2[i]}')
    #     print(f'{i}: {query_performance_avg[i]}, {nsg_query_performance_avg[i]}')
    
    
    # for i in range(n_rknn.shape[0]):
    #     if n_rknn[i] < 200 and in_ds_query_performance[i] < 800:
    #         print(f'{i}: k-occur:{n_rknn[i]}, ndc:{in_ds_query_performance[i]}, ma_dist:{ma_base_dist[i]}')
    
    
    
    # plt.hist(kgraph_query_performance, bins=50, edgecolor='black',color='orange')
    # plt.xlabel('NDC')
    # plt.ylabel('number of queries')
    # plt.tight_layout()
    # plt.title(f'{dataset} query performance distribution')
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-performance-distribution.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-query-performance-distribution.png')
    
    a = kgraph_query_performance
    b = nsg_query_performance_avg
    
    get_density_scatter(a,b)
    print(np.corrcoef(a,b))
    
    # for i in range(len(query_performance_avg)):
    #     if kgraph_query_performance[i] < 5000 and query_performance_avg[i] > 5000: 
    #         print(f'{i}: {kgraph_query_performance[i]}, {query_performance_avg[i]}')