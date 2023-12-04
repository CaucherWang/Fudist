from utils import *
from queue import PriorityQueue
import os
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap


visited_points = None
glanced_points = None

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

def search_with_ep(graph, vec, query, k, KG, efsearch, eps, ndc=0, nhops=0):
    resultSet = PriorityQueue(efsearch)     # len < efSearch
    candidateSet = PriorityQueue()  # unlimited lenth
    visited = set()
    for ep in eps:
        candidateSet.put(node(ep, -euclidean_distance(vec[ep], query)))
        
    while not candidateSet.empty():
        top = candidateSet.get()
        if not resultSet.empty() and -top.dist > resultSet.queue[0].dist:
            break
        nhops +=1
        # visited_points[top.id] += 1
        for nei in graph[top.id][:KG+1]:
            if nei == top.id or nei in visited:
                continue
            visited.add(nei)
            # glanced_points[nei] += 1
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
        
def search(graph, vec, query, k, KG, efsearch, ndc=0, nhops=0):
    eps = select_ep(vec.shape[0])
    return search_with_ep(graph, vec, query, k, KG, efsearch, eps, ndc, nhops)
    
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
     
def get_bidirectional_neighbors(KG, K):
    n = KG.shape[0]
    cnt = 0
    for i in range(n):
        if i % 10000 == 0:
            print(i)
        for j in range(K+1):
            if KG[i][j] == i:
                continue
            if i in KG[KG[i][j]][:K+1]:
                cnt += 1
    return cnt
    
source = './data/'
dataset = 'deep'
efConstruction = 500
M=16
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    graph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')
    hnsw_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}.index')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs')
    KG = 32
    ind_path = os.path.join(source, dataset, f'{dataset}_ind_{KG}.ibin')

    X = read_fvecs(base_path)
    G = read_ivecs(graph_path)
    Q = read_fvecs(query_path)[:10000]
    GT = read_ivecs(GT_path)
    # n_rknn = get_reversed_knn_number(G, KG)
    # write_ibin_simple(ind_path, n_rknn)
    n_rknn = read_ibin_simple(ind_path)
    # print(get_skewness(n_rknn))
    
    # print(get_bidirectional_neighbors(G, KG))
    
    # lengths = compute_lengths(X)
    # write_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'), lengths)
    # lengths = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'))
    
    mean = np.mean(X, axis=0)
    X = X - mean
    norms = np.linalg.norm(X, axis=1)
    print(norms.shape)
    # find the max norm
    max_norm = np.max(norms)
    print(max_norm)
    Q = Q - mean
    Q_norm = Q / max_norm
    q_lengths = np.linalg.norm(Q_norm, axis=1)
    print(q_lengths.shape)
    print(f'max: {np.max(q_lengths)}, min: {np.min(q_lengths)}')
    
    # split_points_datasets = {
    #     'deep':[0.8, 0.95],
    #     'sift':[0.7, 0.9],
    #     'glove1.2m':[0.25, 0.55],
    #     'word2vec':[0.02, 0.25]
    # }
    # split_points = split_points_datasets[dataset]
    # central_set = []
    # middle_set = []
    # shell_set = []
    
    # for i in range(len(q_lengths)):
    #     if q_lengths[i] < split_points[0]:
    #         central_set.append(i)
    #     elif q_lengths[i] < split_points[1]:
    #         middle_set.append(i)
    #     else:
    #         shell_set.append(i)
            
    # print(f"central: {len(central_set)}, middle: {len(middle_set)}, shell: {len(shell_set)}")
    # central_set = central_set[:1000]
    # middle_set = middle_set[:1000]
    # shell_set = shell_set[:1000]
    # three_sets  = [central_set, middle_set, shell_set]
    
    
    # assert KG < G.shape[1]
        
    k = 50
    # # visited_points = np.zeros(X.shape[0], dtype=np.int32)
    # # glanced_points = np.zeros(X.shape[0], dtype=np.int32)
    
    # # efss = [100, 200, 300, 400, 500]
    efss = [160, 440]
    # # efss = [50000, 60000, 80000, 100000]
    for efs in efss:
        print(f'efsearch: {efs}')
        for q_set in three_sets:
            print(f'q_num: {len(q_set)}')
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
            for q_id in q_set:
                q = Q[q_id]
                topk, ndc, nhops = search(G, X, q, k, KG, efs)
                gt = GT[q_id][:k]
                recall = get_recall(topk, gt)
                
                tps, fps, fns = decompose_topk_to_3sets(np.array(topk), np.array(gt))
                sum_tps += len(tps)
                sum_fps += len(fps)
                sum_fns += len(fns)
                
                tp_n_rknn, fp_n_rknn, fn_n_rknn = analyze_results_k_occurrence(tps, fps, fns, n_rknn)
                sum_fp_rknn += fp_n_rknn
                sum_tp_rknn += tp_n_rknn
                sum_fn_rknn += fn_n_rknn
                            
                tp_lengths, fp_lengths, fn_lengths = analyze_results_norm_length(tps, fps, fns, lengths)
                sum_tp_lengths += tp_lengths
                sum_fp_lengths += fp_lengths
                sum_fn_lengths += fn_lengths
                
                idx += 1
                if idx % 1000 == 0:
                    print(idx)
                sum_ndc += ndc
                sum_nhops += nhops
                sum_recall += recall
            # print()
            recall = sum_recall / len(q_set) / k
            ndc = sum_ndc / len(q_set)
            nhops = sum_nhops / len(q_set)
            print(f'recall: {recall}, ndc: {ndc}, nhops: {nhops}')
            print(f'\ttp_rknn: {sum_tp_rknn / sum_tps}, fp_rknn: {sum_fp_rknn / sum_fps}, fn_rknn: {sum_fn_rknn / sum_fns}, tp-fn: {sum_tp_rknn / sum_tps - sum_fn_rknn / sum_fns}')
            print(f'{sum_tp_rknn / sum_tps:.2f} & {sum_fp_rknn / sum_fps:.2f} & {sum_fn_rknn / sum_fns:.2f} & {sum_tp_rknn / sum_tps - sum_fn_rknn / sum_fns:.2f}')
            print(f'\ttp_lengths: {sum_tp_lengths / sum_tps}, fp_lengths: {sum_fp_lengths / sum_fps}, fn_lengths: {sum_fn_lengths / sum_fns}, tp-fn: {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns}')
            print(f'{sum_tp_lengths / sum_tps:.2f} & {sum_fp_lengths / sum_fps:.2f} & {sum_fn_lengths / sum_fns:.2f} & {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns:.2f}')
        
    # import matplotlib.pyplot as plt
    # # plt.plot(n_rknn, visited_points, 'o', color='black', markersize=0.1)
    # # plt.xlabel('indegree of kNN graph')
    # # plt.ylabel('visited_times')
    # # # plt.xlim(0, 1200)
    # # # plt.xscale('log')
    # # # plt.yscale('log')
    # # # plt.title(f'{dataset} indegree correlation')
    # # plt.savefig(f'./figures/{dataset}-visit-hub-correlation.png')
    
    # write_fbin_simple(f'./figures/{dataset}-kgraph-visit-hub-correlation.fbin', glanced_points)
    
    
    # # print(np.corrcoef(n_rknn, glanced_points))
    # # plt.close()
    # plt.plot(n_rknn, glanced_points, 'o', color='black', markersize=0.1)
    # plt.xlabel('Indegree of points')
    # plt.ylabel('Visted times')
    # # plt.xlim(0, 1200)
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.title(f'{dataset} indegree correlation')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}-glance-hub-correlation.png')
    
    # # filter the points with glanced_points > 200
    # filter_pos = np.where(glanced_points > 200)[0]
    # filter_r_knn = n_rknn[filter_pos]
    # print(np.min(filter_r_knn), np.percentile(filter_r_knn, 1), np.percentile(filter_r_knn, 5), np.percentile(filter_r_knn, 50))
    
