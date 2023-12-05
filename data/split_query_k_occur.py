from requests import get
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
    # plt.xlabel('k-occurrence (450)')
    # plt.xlabel('Query difficulty')
    plt.xlabel('page-rank')
    # plt.ylabel('Query performance (recall@50=0.96)')
    plt.ylabel('k-occurrence (500)')
    # plt.ylabel('local intrinsic dimensionality')
    # plt.ylabel('1NN distance')
    # plt.tight_layout()
    # plt.xlim(0, 20)
    plt.ylim(0, 25000)
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
    

source = './data/'
result_source = './results/'
dataset = 'rand100'
idx_postfix = '_plain_1th'
efConstruction = 500
Kbuild = 100
M=16
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
    inter_knn_dist_avg_path = os.path.join(source, dataset, f'{dataset}_inter_knn_dist_avg50.fbin')
    hnsw_ind_path = os.path.join(source, dataset, f'{dataset}_hnsw_ef{efConstruction}_M{M}_ind_{KG}.ibin')
    kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')
    result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_.log{idx_postfix}_unidepth-level')
    ma_dist_path = os.path.join(source, dataset, f'{dataset}_ma_distance.fbin')
    ma_base_dist_path = os.path.join(source, dataset, f'{dataset}_ma_base_distance.fbin')
    kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log')
    in_ds_kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log_in-dataset')
    query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance.log_plain')
    query_performance_log_paths = []
    for i in range(3, 12):
        query_performance_log_paths.append(os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance.log_plain_shuf{i}'))

    # in_ds_kgraph_query_performance = np.array(resolve_performance_variance_log(in_ds_kgraph_query_performance_log_path))
    # kgraph_query_performance = np.array(resolve_performance_variance_log(kgraph_query_performance_log_path))
    # query_performance = np.array(resolve_performance_variance_log(query_performance_log_path))
    # query_performances = [query_performance]
    # for i in range(9):
    #     query_performances.append(np.array(resolve_performance_variance_log(query_performance_log_paths[i])))
    # query_performances = np.array(query_performances)
    
    # query_performance_avg = np.sum(query_performances, axis=0) / len(query_performances)

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
    GT_dist = read_fvecs(GT_dist_path)
    KGraph = read_ivecs(kgraph_path)
    lengths = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'))
    # q_lengths = get_query_length(X, Q)
    n_rknn = read_ibin_simple(ind_path)
    # n_rknn2 = read_ibin_simple(ind_path2)
    # indegree = get_indegree_list(G)
    # write_ibin_simple(hnsw_ind_path, indegree)
    # indegree = read_ibin_simple(hnsw_ind_path)
    # revG = get_reversed_graph_list(G)
    # print(np.where(GT == 885271))
    
    # KGraph_clean = clean_kgraph(KGraph)
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs_clean'), KGraph_clean)
    KGraph = read_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs_clean'))
    # csr_graph = adjacency_list_to_csr(KGraph)
    alpha = 0.95
    # pagerank = compute_pagerank2(KGraph, alpha)
    # write_fbin_simple(os.path.join(source, dataset, f'{dataset}_pagerank.fbin_{alpha}'), pagerank)
    pagerank = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_pagerank.fbin_{alpha}'))
    # plt.hist(pagerank, bins=50, edgecolor='black', linewidth=1.2)
    # plt.xlabel('pagerank')
    # plt.ylabel('number of points')
    # plt.title(f'{dataset} pagerank', fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-pagerank.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-pagerank.png')
    
    # get_density_scatter(pagerank, in_ds_kgraph_query_performance)
    
    # plt.hist(n_rknn, bins=80, edgecolor='black', linewidth=1.2)
    # plt.xlabel('k-occurrence')
    # plt.xticks(np.arange(0, 5000, 500), fontsize=12)
    # plt.ylabel('number of points')
    # plt.title(f'{dataset} k-occurrence', fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-k-occurrence.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-k-occurrence.png')
    
    # print(np.corrcoef(n_rknn, n_rknn2))
    # get_density_scatter(n_rknn, n_rknn2)
    
    
    q_k_occurs = []
    for i in range(Q.shape[0]):
        gt = GT[i][:50]
        sum_k_occur = 0
        for j in range(50):
            sum_k_occur += n_rknn[gt[j]]
        q_k_occurs.append(sum_k_occur / 50)
    q_k_occurs = np.array(q_k_occurs)
    # find the indexes of 200 minimum k_occurs
    k_occurs_indexes = np.argsort(q_k_occurs)
    npart = 10
    
    q_pagerank = []
    for i in range(Q.shape[0]):
        gt = GT[i][:50]
        sum_pagerank = 0
        for j in range(50):
            sum_pagerank += pagerank[gt[j]]
        q_pagerank.append(sum_pagerank)
    q_pagerank = np.array(q_pagerank)
    
    q_over_k_occurs = []
    for j in range(Q.shape[0]):
        gt = GT[j][:99]
        sum_k_occur = 0
        for j in range(99):
            if n_rknn[gt[j]] < 300:
                sum_k_occur += 1
        q_over_k_occurs.append(sum_k_occur / 99)
    q_over_k_occurs = np.array(q_over_k_occurs)
    # find the indexes of 200 minimum k_occurs
    over_k_occurs_indexes = np.argsort(q_k_occurs)
    
    # plt.hist(q_over_k_occurs, bins=50, edgecolor='black', linewidth=1.2)
    # plt.xlabel('q over k-occurrence')
    # plt.ylabel('number of queries')
    # plt.title(f'{dataset} q over k-occurrence', fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-q-over-k-occurrence.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-q-over-k-occurrence.png')
    
    # with open(os.path.join(result_source, dataset, f'{dataset}_query4023.csv'), 'w') as f:
    #     for i in range(50):
    #         f.write(f'0, {GT[4023][i]}, {GT_dist[4023][i]}\n')
    #         f.write(f'{GT[4023][i]}, 0, {GT_dist[4023][i]}\n')
    
    def clean_sub_graph(subgraph, points):
        points = set(points)
        ret = []
        for i in range(len(subgraph)):
            neighbors = []
            for nei in subgraph[i]:
                if nei in points:
                    neighbors.append(nei)
            ret.append(neighbors)
        return ret
    
    special_qid = 4023
    knn = GT[special_qid][:50]
    for i in [100, 200, 300, 400, 500]:
        points = GT[special_qid][:i]
        subgraph = KGraph[points]
        subgraph = clean_sub_graph(subgraph, points)
        
        
        

    
    dist1nn = GT_dist[:, 0]
    # replace zeros with 1e-5
    # dist1nn = np.array([1e-4 if x <1e-4 else x for x in dist1nn])
    exp_dist1nn = np.exp(dist1nn)
    
    distknn = GT_dist[:, 99]
    dist_range_knn = distknn - dist1nn
    
    # inter_knn_dist = []
    # for i in range(Q.shape[0]):
    #     inter_knn_dist.append(get_inter_knn_dist(X, GT[i][:50]))
    # write_fbin_simple(inter_knn_dist_avg_path, inter_knn_dist)
    inter_knn_dist = read_fbin_simple(inter_knn_dist_avg_path)
    
    q_lids = get_lids(GT_dist[:, :50], 50)
    # q_lids_plan_exp = get_lids_plan_exp(GT_dist[:, :50], 50)
    q_lids_plan_linear = get_lids_plan_linear(GT_dist[:, :50], 50)
    query_ma_dist = get_mahalanobis_distance(X, Q)
    # base_ma_dist = read_fbin_simple(ma_base_dist_path)
    # plt.hist(base_ma_dist, bins=50, edgecolor='black', linewidth=1.2)
    # plt.xlabel('Mahalanobis distance')
    # plt.ylabel('number of points')
    # plt.title(f'{dataset} Mahalanobis distance distribution')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-mahalanobis-distance-distribution.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-mahalanobis-distance-distribution.png')
    # lid_indexes = np.argsort(q_lids)
    
    
    special_qid = 4023
    missed_knn = [885271,918350,756805,558024,667395,119986,471653,219295,471048]
    # missed_knn = [785259,706403,235791,253119,505724,574760,732116,556718]
    for _ in missed_knn:
        print(f'{_}: {np.where(GT == _)}')

    # missed_knn_index = [44,32,29,15,14,13,12,10,9]
    gt_special = GT[special_qid, :50]
    gt_special_vecs = X[gt_special]
    # special_q2gt_vecs =  gt_special_vecs - Q[special_qid]
    # cosines = []
    # for i in range(50):
    #     tmp = []
    #     for j in range(50):
    #         tmp.append(get_cosine(special_q2gt_vecs[i], special_q2gt_vecs[j]))
    #     cosines.append(tmp)
    # avg_cosine = np.average(np.array(cosines), axis=1)
    # print(avg_cosine)
    # not full rank: too few samples
    # ma_dist = get_mahalanobis_distance(gt_special_vecs, gt_special_vecs[0])
    # print(np.average(ma_dist))
    # print(ma_dist[missed_knn_index])
    # subG = KGraph[gt_special]
    # graph2csv(subG, gt_special, X, os.path.join(source, dataset, f'{dataset}_subgraph.csv'))
    # gt_special = [x for x in gt_special if x not in missed_knn]
    # gt_special_ma_dist = base_ma_dist[gt_special]
    # missed_knn_ma_dist = base_ma_dist[missed_knn]
    # print(query_ma_dist[special_qid], np.average(gt_special_ma_dist), np.average(missed_knn_ma_dist)) 
    
    # for i in range(200):
    #     print(f'query {i}:', end='')
    #     array = []
    #     for j in range(8):
    #         array.append(query_performances[j][i])
    #     array = np.array(array)
    #     stdev = np.std(array)
    #     print(f'stdev: {stdev}; ', end='')
    #     pprint(array)
    
    # plt.hist(query_ma_dist, bins=50, edgecolor='black', linewidth=1.2)
    # plt.xlabel('query ma dist')
    # plt.ylabel('number of queries')
    # plt.title(f'{dataset} ma dist', fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-hist-ma-dist.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-hist-ma-dist.png')

    
    # new_metric = (q_k_occurs * dist_range_knn * np.exp(0.05*exp_dist1nn)) / (exp_dist1nn)
    # new_metric = q_lids * (1 + q_over_k_occurs)
    new_metric = q_lids_plan_linear
    # new_metric = ((dist1nn / np.exp(2*dist_range_knn)) + np.average(GT_dist[:, :50], axis=1)) * (1 + q_over_k_occurs)
    # new_metric = np.zeros(Q.shape[0])
    # for i in range(Q.shape[0]):
    #     if dist1nn[i] < 0.3:
    #         new_metric[i] = (q_k_occurs[i] * (np.exp(dist_range_knn[i])) ) / (exp_dist1nn[i])
    #     elif dist1nn[i] < 0.55:
    #         new_metric[i] = (q_k_occurs[i] * (np.exp(dist_range_knn[i] ))) / (exp_dist1nn[i])
    #     else:
    #         new_metric[i] = (q_k_occurs[i] * (np.exp(dist_range_knn[i]))) / (exp_dist1nn[i])
    # new_metric = (q_k_occurs * (np.exp(dist_range_knn * 0.7))) / (exp_dist1nn)
    # replace the elements > 1m with 1m
    # new_metric = np.array([1e5 if x > 1e5 else x for x in new_metric])
    # sorted_new_metric = np.sort(new_metric)
    # new_metric_indexes = np.argsort(new_metric)
    
    # get_density_scatter(new_metric, kgraph_query_performance)
    
    selected_query_index = []
    
    for i in range(Q.shape[0]):
        # if new_metric[i] <1 and kgraph_query_performance[i] > 9000:
        if kgraph_query_performance[i] > 30000:
            selected_query_index.append(i)
            # print(f'{i}: new metric: {new_metric[i]}, ndc: {kgraph_query_performance[i]}, inter_knn_dist:{inter_knn_dist[i]}, lid: {q_lids[i]}, k_occur: {q_k_occurs[i]}, over_k_occur: {q_over_k_occurs[i]}, ma dist: {query_ma_dist[i]}, 1NN dist: {dist1nn[i]}, dist range knn: {dist_range_knn[i]}, {(dist1nn[i] / np.exp(2*dist_range_knn[i]))}, {np.average(GT_dist[i])}') 

    # find the position of 4192 and 4023 in selected query index
    # print(f'4192: {selected_query_index.index(4192)}')
    # print(f'4023: {selected_query_index.index(4023)}')
    
    i = 4023
    gt_i = GT[i][:50]
    gt_vecs = X[gt_i]
    gt_page_rank = pagerank[gt_i]
    # print(np.average(gt_page_rank))
    # # cluster gt_vecs
    # missed_knn_index = np.array([44,32,29,15,14,13,12,10,9]) - 1
    # from sklearn.cluster import KMeans
    # for k in range(2, 10):
    #     kmeans = KMeans(n_clusters=k, random_state=0).fit(gt_vecs)
    #     # print the quliaty of clusters
    #     print(f'k={k}: {kmeans.inertia_}')
    #     print(kmeans.labels_)
    #     print(kmeans.labels_[missed_knn_index])
    #     centers = kmeans.cluster_centers_
    #     dist2center = []
    #     for i in range(50):
    #         dist2center.append(euclidean_distance(gt_vecs[i], centers[kmeans.labels_[i]]))
    #     avg_dist2center = []
    #     for i in range(k):
    #         avg_dist2center.append(np.average(np.array(dist2center)[np.where(kmeans.labels_ == i)]))
    #     print(avg_dist2center)
    #     print(np.array(dist2center)[missed_knn_index])
    
    inter_dist = []
    for i in range(50):
        tmp = []
        for j in range(50):
            tmp.append(euclidean_distance(gt_vecs[i], gt_vecs[j]))
        inter_dist.append(tmp)
        
    gt_knn_dist = []
    for i in range(50):
        gt_knn_dist.append(euclidean_distance(X[gt_i[i]],X[KGraph[gt_i[i]][98]]))
        
    inter_relative_dist = []
    for i in range(50):
        tmp = []
        for j in range(50):
            tmp.append(inter_dist[i][j] / gt_knn_dist[i])
        inter_relative_dist.append(np.average(tmp))
    
       
    gaps = []
    for i in range(50):
        tmp = []
        for j in range(50):
            gap = inter_dist[i][j] - gt_knn_dist[i] - gt_knn_dist[j]
            gap = max(0, gap)
            tmp.append(gap)
        gaps.append(np.average(tmp))
            
        
    # print(inter_dist)
    

    # # sort selected query index by ndc
    # selected_query_index = np.array(selected_query_index)
    # selected_query_index = selected_query_index[np.argsort(kgraph_query_performance[selected_query_index])]
    # for i in selected_query_index:
    #     print(f'{i}: new metric: {new_metric[i]}, ndc: {kgraph_query_performance[i]}, inter_knn_dist:{inter_knn_dist[i]}, lid: {q_lids[i]}, pagerank:{q_pagerank[i]}, k_occur: {q_k_occurs[i]}, over_k_occur: {q_over_k_occurs[i]}, ma dist: {query_ma_dist[i]}, 1NN dist: {dist1nn[i]}, dist range knn: {dist_range_knn[i]}') 

    # print(len(selected_query_index))
    # selected_query_index = np.array(selected_query_index)
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_hardest'), Q[selected_query_index])
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_hardest'), GT[selected_query_index])
    
    # query_performance_avg1 = (query_performances[3] +query_performances[2]+query_performances[6]+query_performances[7] +query_performances[1]) / 5.0
    # query_performance_avg2 = (query_performances[4] + query_performances[5]+query_performances[8]+query_performances[9] +query_performances[2]) / 5.0
    
    # print(np.corrcoef(query_performance_avg1, query_performance_avg2))
    # get_density_scatter(query_performance_avg1, query_performance_avg2)
    
    # get_density_scatter(query_performances[4], query_performances[3])
    
    # plt.hist(query_performance, bins=50, edgecolor='black', linewidth=1.2)
    # plt.xlabel('NDC of queries (recall@50=0.96)')
    # plt.ylabel('number of queries')
    # plt.title(f'{dataset} query performance', fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-performance.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-query-performance.png')
    
    # log_new_metric = np.log2(new_metric)
    # plt.hist(log_new_metric, bins=50, edgecolor='black', linewidth=1.2)
    # plt.xlabel('difficulty of queries')
    # plt.ylabel('number of queries')
    # plt.title(f'{dataset} query performance', fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-difficulty-hist.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-difficulty-hist.png')
    
    
    
    # ma_dist = get_mahalanobis_distance(X, Q)
    # write_fbin_simple(ma_dist_path, ma_dist)
    # ma_dist = read_fbin_simple(ma_dist_path)
    
    # for i in range(Q.shape[0]):
    #     if new_metric[i] < 200 and query_performance[i] < 1200:
    #     # if new_metric[i] > 10000:
    #         q_id = i
    #         lid = q_lids[q_id]
    #         gt_dist_min = GT_dist[q_id][0]
    #         gt_dist_max = GT_dist[q_id][-1]
    #         gt_dist_range = gt_dist_max - gt_dist_min
    #         ma_dist = query_ma_dist[q_id]
    #         print(f'{i}: q_id: {q_id}, ndc:{query_performance[i]}, difficulty:{new_metric[i]}, lid: {lid}, k_occur:{q_k_occurs[q_id]}, ma dist:{ma_dist}, gt_dist_min: {gt_dist_min}, gt_dist_max: {gt_dist_max}, gt_dist_range: {gt_dist_max - gt_dist_min}')

    
    easy_q_indexes = []
    high_k_occur_q_indexes = []
    contrastive_q_indexes = []
    k_occurs = []
    # for i in range(2000):
    #     q_id = k_occurs_indexes[i]
    #     lid = q_lids[q_id]
    #     gt_dist_min = GT_dist[q_id][0]
    #     gt_dist_max = GT_dist[q_id][-1]
    #     if lid < 10:
    #         easy_q_indexes.append(q_id)
    #     elif len(contrastive_q_indexes) < len(easy_q_indexes) and lid > 16:
    #         contrastive_q_indexes.append(q_id)
        # k_occurs.append(n_rknn[q_id])

    # for i in range(2000):
    #     q_id = lid_indexes[-i-1]
    #     lid = q_lids[q_id]
    #     gt_dist_min = GT_dist[q_id][0]
    #     gt_dist_max = GT_dist[q_id][-1]
    #     if q_k_occurs[q_id] > 60:
    #         easy_q_indexes.append(q_id)
    #     elif len(contrastive_q_indexes) < len(easy_q_indexes) :
    #         contrastive_q_indexes.append(q_id)


    # plt.hist(k_occurs, bins=50, edgecolor='black', linewidth=1.2)
    # plt.xlabel('k_occurs for small lid, close to 1NN')
    # plt.ylabel('number of queries')
    # plt.title(f'{dataset} k_occurs for small lid, close to 1NN', fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-k_occurs-for-small-lid-close-to-1NN.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-k_occurs-for-small-lid-close-to-1NN.png')

    # compact_queries = Q[easy_q_indexes]
    # compact_ground_truth = GT[easy_q_indexes]
    # k_occurs = []
    # lids = []
    # for i in range(len(easy_q_indexes)):
    #     q_id = easy_q_indexes[i]
    #     k_occurs.append(q_k_occurs[q_id])
    #     lids.append(q_lids[q_id])
    # k_occurs = np.array(k_occurs)
    # lids = np.array(lids)
    # print(f'low k_occur: avg k-occur: {np.average(k_occurs)}, min-k-occur:{np.min(k_occurs)}, max k-occur: {np.max(k_occurs)}, avg lid: {np.average(lids)}')
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_compact'), compact_ground_truth)
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_compact'), compact_queries)
    # print(len(compact_queries))
    
    # contrastive_queries = Q[contrastive_q_indexes]
    # contrastive_ground_truth = GT[contrastive_q_indexes]
    # k_occurs = []
    # lids = []
    # for i in range(len(contrastive_q_indexes)):
    #     q_id = contrastive_q_indexes[i]
    #     k_occurs.append(q_k_occurs[q_id])
    #     lids.append(q_lids[q_id])
    # k_occurs = np.array(k_occurs)
    # lids = np.array(lids)
    # print(f'low k_occur: avg k-occur: {np.average(k_occurs)}, min-k-occur:{np.min(k_occurs)}, max k-occur: {np.max(k_occurs)}, avg lid: {np.average(lids)}')

    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_contrastive'), contrastive_ground_truth)
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_contrastive'), contrastive_queries)
    # print(len(contrastive_q_indexes))

    
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


    # plt.hist(q_k_occurs / np.max(q_k_occurs), bins=100, edgecolor='black')
    # plt.xlabel('k_occurs')
    # plt.ylabel('number of queries')
    # plt.title(f'{dataset} k_occurs distribution')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k_occurs-distribution.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-query-k_occurs-distribution.png')
    
    # query_splits = []
    # q_index_splits = []
    # r = q_k_occurs.max() - q_k_occurs.min()
    # part_size = r / npart
    # start = 0
    # for t in range(npart):
    #     part_max = q_k_occurs.min() + (t+1) * part_size
    #     if t == npart - 1:
    #         part_max = q_k_occurs.max() + 1
    #     print(f'from {q_k_occurs.min() + t * part_size} to {part_max}')
    #     pre_start = start
    #     while start < Q.shape[0] and q_k_occurs[k_occurs_indexes[start]] < part_max:
    #         start += 1
    #     qindexes = k_occurs_indexes[pre_start: start]
    #     q_index_splits.append(qindexes)
    #     query_splits.append(Q[qindexes])
    #     qpath = os.path.join(source, dataset, f'{dataset}_query.fvecs_unidepth-level{t}')
    #     gtpath = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_unidepth-level{t}')
    #     write_fvecs(qpath, Q[qindexes])
    #     write_ivecs(gtpath, GT[qindexes])

    
    # part_size = Q.shape[0] // npart
    # query_splits = []
    # for i in range(npart):
    #     qindexes = new_metric_indexes[i*part_size:(i+1)*part_size]
    #     query_splits.append(qindexes)
    #     # qpath = os.path.join(source, dataset, f'{dataset}_query.fvecs_new_metric_level{i}')
    #     # gtpath = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_new_metric_level{i}')
    #     # write_fvecs(qpath, Q[qindexes])
    #     # write_ivecs(gtpath, GT[qindexes])
        
    # cur_split = query_splits[0]
    # for i in range(len(cur_split)):
    #     q_id = cur_split[i]
    #     lid = q_lids[q_id]
    #     gt_dist_min = GT_dist[q_id][0]
    #     gt_dist_max = GT_dist[q_id][-1]
    #     gt_dist_range = gt_dist_max - gt_dist_min
    #     ma_dist = query_ma_dist[q_id]
    #     print(f'{i}: q_id: {q_id}, new_metric:{q_k_occurs[q_id] / gt_dist_min}, perform:{performance[i]}, k_occur:{q_k_occurs[q_id]}, ma dist:{ma_dist}, gt_dist_min: {gt_dist_min}, gt_dist_max: {gt_dist_max}, gt_dist_range: {gt_dist_max - gt_dist_min}')

        
    
    
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

    
