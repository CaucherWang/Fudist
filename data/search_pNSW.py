from utils import *
from queue import PriorityQueue
import os

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
    
def prune_NSW(G, X, K):
    D = compute_pairwiese_distance_list(X, G)
    for i in range(len(D)):
        if i % 100000 == 0:
            print(i)
        sorted_dist = sorted(D[i])
        smallest_indices = [D[i].index(sorted_dist[j]) for j in range(min(K, len(D[i])))]
        G[i] = [G[i][j] for j in smallest_indices]
        if(len(G[i]) < K):
            G[i] += [-1] * (K - len(G[i]))
        G[i] = np.array(G[i], dtype='int32')
    return np.array(G)
    
def do_expr(X, Q, GT, k, n_rknn, lengths, indegree):
    
    # efss = [100, 200, 300, 400, 500]
    efss = [60,80, 100, 150, 200, 300, 400, 500, 600, 750, 1000, 1500, 2000]
    # efss = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 5200, 5400, 5600, 5800, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000, 12000, 15000, 18000, 20000, 25000, 30000, 35000, 40000]
    # efss = [50000, 60000, 80000, 100000]
    for efs in efss:
        visited_points = np.zeros(X.shape[0], dtype=np.int32)
        glanced_points = np.zeros(X.shape[0], dtype=np.int32)
        print(f'efsearch: {efs}', end='\t')
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
            topk, ndc, nhops = search(G, X, q, k,  efs)
            gt = GT[idx][:k]
            recall = get_recall(topk, gt)
            
            tps, fps, fns = decompose_topk_to_3sets(np.array(topk), np.array(gt))
            sum_tps += len(tps)
            sum_fps += len(fps)
            sum_fns += len(fns)
            
            tp_n_rknn, fp_n_rknn, fn_n_rknn = analyze_results_k_occurrence(tps, fps, fns, n_rknn)
            sum_fp_rknn += fp_n_rknn
            sum_tp_rknn += tp_n_rknn
            sum_fn_rknn += fn_n_rknn
            
            tp_indegree, fp_indegree, fn_indegree = analyze_results_indegree(tps, fps, fns, indegree)
            sum_tp_indegree += tp_indegree
            sum_fp_indegree += fp_indegree
            sum_fn_indegree += fn_indegree
                        
            tp_lengths, fp_lengths, fn_lengths = analyze_results_norm_length(tps, fps, fns, lengths)
            sum_tp_lengths += tp_lengths
            sum_fp_lengths += fp_lengths
            sum_fn_lengths += fn_lengths
            
            idx += 1
            # if idx % 1000 == 0:
            #     print(idx)
            sum_ndc += ndc
            sum_nhops += nhops
            sum_recall += recall
        # print()
        recall = sum_recall / Q.shape[0] / k
        ndc = sum_ndc / Q.shape[0]
        nhops = sum_nhops / Q.shape[0]
        print(f'recall: {recall}, ndc: {ndc}, nhops: {nhops}')
        print(f'\ttp_rknn: {sum_tp_rknn / sum_tps}, fp_rknn: {sum_fp_rknn / sum_fps}, fn_rknn: {sum_fn_rknn / sum_fns}, tp-fn: {sum_tp_rknn / sum_tps - sum_fn_rknn / sum_fns}')
        print(f'{sum_tp_rknn / sum_tps:.2f} & {sum_fp_rknn / sum_fps:.2f} & {sum_fn_rknn / sum_fns:.2f} & {sum_tp_rknn / sum_tps - sum_fn_rknn / sum_fns:.2f}')
        print(f'\ttp_indegree: {sum_tp_indegree / sum_tps}, fp_indegree: {sum_fp_indegree / sum_fps}, fn_indegree: {sum_fn_indegree / sum_fns}, tp-fn: {sum_tp_indegree / sum_tps - sum_fn_indegree / sum_fns}')
        print(f'{sum_tp_indegree / sum_tps:.2f} & {sum_fp_indegree / sum_fps:.2f} & {sum_fn_indegree / sum_fns:.2f} & {sum_tp_indegree / sum_tps - sum_fn_indegree / sum_fns:.2f}')
        print(f'\ttp_lengths: {sum_tp_lengths / sum_tps}, fp_lengths: {sum_fp_lengths / sum_fps}, fn_lengths: {sum_fn_lengths / sum_fns}, tp-fn: {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns}')
        print(f'{sum_tp_lengths / sum_tps:.2f} & {sum_fp_lengths / sum_fps:.2f} & {sum_fn_lengths / sum_fns:.2f} & {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns:.2f}')
        # write_fbin_simple(f'./figures/{dataset}-pNSW-visit-hub-correlation-ef{efs}.fbin', visited_points)
        # write_fbin_simple(f'./figures/{dataset}-pNSW-glance-hub-correlation-ef{efs}.fbin', glanced_points)
    
source = './data/'
dataset = 'sift'
efConss = {
    'deep': 500,
    "sift": 500,
    'rand200':500,
    'gauss200':500,
    "glove1.2m":500,
    "word2vec":500
}
query_postfix = ''
efConstruction = efConss[dataset]
Kbuild = 16
KG = 32
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    graph_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_K{Kbuild}.nsw.index')
    Kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')
    pruned_graph_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_K{Kbuild}_pruned.nsw.index')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs{query_postfix}')
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs{query_postfix}')
    ind_path = os.path.join(source, dataset, f'{dataset}_ind_{KG}.ibin')
    nsw_ind_path = os.path.join(source, dataset, f'{dataset}_nsw_ind_{KG}.ibin')
    pruned_nsw_ind_path = os.path.join(source, dataset, f'{dataset}_pruned_nsw_ef{efConstruction}_K{Kbuild}_ind_{KG}.ibin')

    X = read_fvecs(base_path)
    # NSW = read_nsw(graph_path, X.shape[0])
    Q = read_fvecs(query_path)[:2000]
    GT = read_ivecs(GT_path)
    # G = prune_NSW(NSW, X, KG)
    # write_ivecs(pruned_graph_path, G)
    G = read_ivecs(pruned_graph_path)
    KGraph = read_ivecs(Kgraph_path)
    lengths = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'))
    
    # print(get_graph_quality(G, KGraph, KG))
    # print(get_graph_quality_list(NSW, KGraph, KG))
    
    # print(count_edges(G))
    
    n_rknn = read_ibin_simple(ind_path)
    
    # indegree = get_indegree(G)
    # write_ibin_simple(pruned_nsw_ind_path, indegree)
    indegree = read_ibin_simple(pruned_nsw_ind_path)
    # get_indegree_hist(indegree, dataset, 'pnsw')
    # insertion_order = np.arange(indegree.shape[0])[::-1]
    # print(np.corrcoef(n_rknn, indegree))
    # print(np.corrcoef(insertion_order, indegree))
    
    # # find the elements with the largest indegree
    # get_outlink_density(indegree, G, X.shape[0])
        
    
    # print(get_skewness(indegree))
    # print(np.corrcoef(np.arange(X.shape[0]), indegree))
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
    # plt.savefig(f'./figures/{dataset}-nsw-indegree-correlation.png')

    
        
    k = 50
    do_expr(X, Q, GT, k, n_rknn, lengths, indegree)

    
        
    # glanced_points = read_ibin_simple(f'./figures/{dataset}-pnsw-glance-hub-correlation.fbin')
    # # remove the max
    # max_pos = np.argmax(glanced_points)
    # glanced_points = np.delete(glanced_points, max_pos)
    # indegree = np.delete(indegree, max_pos)
        
    
    
    # plt.plot(indegree, visited_points, 'o', color='black', markersize=0.1)
    # plt.xlabel('indegree of pNSW')
    # plt.ylabel('visited_times')
    # # plt.xlim(0, 1200)
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.title(f'{dataset} indegree correlation')
    # plt.savefig(f'./figures/{dataset}-pnsw-visit-hub-correlation.png')
    
    # print(np.corrcoef(n_rknn, glanced_points))
    # plt.close()
    # plt.plot(indegree, glanced_points, 'o', color='black', markersize=0.1)
    # plt.xlabel('Indegree of points')
    # plt.ylabel('Visted times')
    # # plt.xlim(0, 1200)
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.title(f'{dataset} indegree correlation')
    # plt.savefig(f'./figures/{dataset}-pnsw-glance-hub-correlation.png')
    
    # # filter the points with glanced_points > 200
    # filter_pos = np.where(glanced_points > 200)[0]
    # filter_r_knn = n_rknn[filter_pos]
    # print(np.min(filter_r_knn), np.percentile(filter_r_knn, 1), np.percentile(filter_r_knn, 5), np.percentile(filter_r_knn, 50))
