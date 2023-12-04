from turtle import circle
from requests import get
from sklearn.utils import deprecated
from utils import *
from queue import PriorityQueue
import os
from unionfind import UnionFind
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
    plt.xlabel('ME_GREEDY')
    # plt.ylabel('Query performance (recall@50=0.96)')
    plt.ylabel('NDC')
    # plt.ylabel('local intrinsic dimensionality')
    # plt.ylabel('1NN distance')
    # plt.tight_layout()
    plt.xlim(0, 1600)
    # plt.ylim(0, 25000)
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
    
    
def update_reachable(G, X, V, unsettled):
    '''
    Only points in Y are considered.
    Check whether any two points in X are reachable on G.
    '''
    set_X = set(X)
    set_V = set(V)
    successed = set()
    for node in unsettled:
        visited = set()
        visited_X = set()
        stack = [node]
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            if vertex in set_X:
                visited_X.add(vertex)
                
            for v in G[vertex]:
                if v in set_X:
                    visited_X.add(v)
                if v not in visited and v in set_V:
                    stack.append(v)
            
            if len(visited_X) == len(set_X):
                successed.add(node)
                break

        if len(visited_X) < len(set_X):
            unsettled -= successed
            return
            
    unsettled.clear()

def get_reachable(G, q, V):
    '''
    Only points in Y are considered.
    Check whether any two points in X are reachable on G.
    '''
    set_V = set(V)
    visited = set()
    visited_V = set()
    visited_V.add(q)
    stack = [q]
    while stack:
        vertex = stack.pop()
        visited.add(vertex)
            
        for v in G[vertex]:
            if v in set_V:
                visited_V.add(v)
                if v not in visited:
                    stack.append(v)
        
        if len(visited_V) == len(set_V):
            break

    return visited_V

# deprecated
def get_me_greedy(G, KNN, GT_list, revG):
    K = KNN.shape[0]
    reachable_tbl = np.zeros((GT_list.shape[0], GT_list.shape[0]), dtype=np.int32)
    id_to_index = {}
    for i in range(GT_list.shape[0]):
        id_to_index[GT_list[i]] = i
    
    def init_reachable_tbl():
        V = set(KNN)
        for i in range(len(KNN)):
            # check whether other points in KNN is reachable
            visited_by_iNN = get_reachable(G, KNN[i], V)
            for point in visited_by_iNN:
                reachable_tbl[i][id_to_index[point]] = 1
            
    def check_settled():
        success = set()
        for index in unsettled_knn:
            flag = True
            for i in range(K):
                if reachable_tbl[index][i] == 0:
                    flag = False
                    break
            if flag:
                success.add(index)
        for index in success:
            unsettled_knn.remove(index)
            
    init_reachable_tbl()
    unsettled_knn = set(np.arange(K))
    check_settled()
    
    # neighbors_set = {}
    # for point in KNN:
    #     neighbors_set[point] = set(G[point])
    
    cur_index = len(KNN)
    Y = set(KNN)
    while len(unsettled_knn) > 0:
        new_point = GT_list[cur_index]
        Y.add(new_point)
        # set new_point's reachable set
        out_neighbors = G[new_point]
        # neighbors_set[new_point] = set(out_neighbors)
        reachable_set = set()
        reachable_set.add(cur_index)
        for out_neighbor in out_neighbors:
            if out_neighbor not in Y:
                continue
            reachable_tbl[cur_index][id_to_index[out_neighbor]] = 1
            reachable_set.add(id_to_index[out_neighbor])
            for i in range(K):
                if reachable_tbl[cur_index][i] == 0 and reachable_tbl[id_to_index[out_neighbor]][i] == 1:
                    reachable_tbl[cur_index][i] = 1
                    reachable_set.add(i)
                                
        # find all the points that can reach new_point
        in_neigbors_index = set()
        for point in revG[new_point]:
            in_neigbors_index.add(id_to_index[point])
            for i in range(cur_index):
                if reachable_tbl[i][id_to_index[point]] == 1:
                    in_neigbors_index.add(i)
        for index in in_neigbors_index:
            for reach in reachable_set:
                reachable_tbl[index][reach] = 1
        
        check_settled()
        cur_index += 1
    
    if len(unsettled_knn) > 0:
        return GT_list.shape[0]
    return cur_index
            
def get_me_greedy_usg(G, KNN, GT_list, revG):
    K = KNN.shape[0]
    assert np.allclose(KNN, GT_list[:K])

    uf = UnionFind(np.arange(K))
    usg = {}    # key is index of point, value is the out-neighering RSCCs
    for i in range(K):
        usg[i] = set()
    id_to_index = {}
    for i in range(GT_list.shape[0]):
        id_to_index[GT_list[i]] = i
    
    def init_usg():
        V = set(KNN)
        for i in range(len(KNN)):
            # check whether other points in KNN is reachable
            visited_by_iNN = get_reachable(G, KNN[i], V)
            for point in visited_by_iNN:
                index = id_to_index[point]
                if i == index:
                    continue
                usg[i].add(index)
    
    def find_circle_in_usg():
        # sort the points by indegree
        visited = set()
        circle = []
        
        def dfs(node, path, path_indexes):
            nonlocal circle
            visited.add(node)
            path.append(node)
            path_indexes[node] = len(path) - 1

            neighbor_set = usg[node].copy()
            for neighbor in neighbor_set:
                neighbor_root  = uf.find(neighbor)  # neighbors of usg may contain outdated points
                if neighbor_root != neighbor:
                    usg[node].remove(neighbor)
                if neighbor_root in path_indexes:
                    cycle_start = path_indexes[neighbor_root]
                    cur_len = len(path) - cycle_start
                    if cur_len > len(circle):
                        circle = path[cycle_start:]
                elif neighbor_root not in visited:
                    dfs(neighbor_root, path, path_indexes)

            
            path_indexes.pop(node)
            path.pop()
                    
        for node in usg:
            if node not in visited:
                dfs(node, [], {})
        return circle
                
    def union_usg(circle):
        # only update the rscc involved, other rsccs are not affected
        # merge the circle
        first = circle[0]
        out_neighbors = usg[first]
        for i in range(1, len(circle)):
            uf.union(circle[i], first)  # who is the root depends on the size
            out_neighbors = out_neighbors.union(usg[circle[i]])
        
        out_neighbors = set([uf.find(x) for x in out_neighbors])
        root = uf.find(first)
        if root in out_neighbors:
            out_neighbors.remove(root)
        for i in range(len(circle)):
            if root != circle[i]:
                del usg[circle[i]]
        usg[root] = out_neighbors
        
                
    init_usg()
    while True:
        circle = find_circle_in_usg()
        assert len(circle) != 1
        if len(circle) <= 0:
            break
        union_usg(circle)
    
    cur_knn_components = set()
    for i in range(K):
        cur_knn_components.add(uf.find(i))
    if len(cur_knn_components) == 1:
        return K
    
    Y = set(KNN)
        
    for i in range(K, GT_list.shape[0]):
        new_point = GT_list[i]
        out_neighbors = set(G[new_point]).intersection(Y)
        in_neighbors = set(revG[new_point]).intersection(Y)
        Y.add(new_point)
        out_neighbor_roots = set([uf.find(id_to_index[x]) for x in out_neighbors])
        in_neighbor_roots = set([uf.find(id_to_index[x]) for x in in_neighbors])
        uf.add(i)
        usg[i] = out_neighbor_roots.copy()
        intersect = out_neighbor_roots.intersection(in_neighbor_roots)
        if len(intersect) > 0:
            union_usg(list(intersect))
        i_root = uf.find(i)
        for in_neighbor_root in in_neighbor_roots:
            if in_neighbor_root in usg:
                usg[in_neighbor_root].add(i_root)
        
        while True:
            cur_knn_components = set([uf.find(r) for r in cur_knn_components])
            if len(cur_knn_components) == 1:
                return i + 1

            circle = find_circle_in_usg()
            if len(circle) <= 0:
                break
            union_usg(circle)

    return GT_list.shape[0]
  

source = './data/'
result_source = './results/'
dataset = 'deep'
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
    reversed_kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs_reversed')
    result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_.log{idx_postfix}_unidepth-level')
    ma_dist_path = os.path.join(source, dataset, f'{dataset}_ma_distance.fbin')
    ma_base_dist_path = os.path.join(source, dataset, f'{dataset}_ma_base_distance.fbin')
    me_greedy_path = os.path.join(source, dataset, f'{dataset}_me_greedy.ibin')
    me_greedy_path_opt = os.path.join(source, dataset, f'{dataset}_me_greedy.ibin_usg')
    kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log')
    in_ds_kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log_in-dataset')
    query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance.log_plain')
    query_performance_log_paths = []
    for i in range(3, 12):
        query_performance_log_paths.append(os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance.log_plain_shuf{i}'))

    in_ds_kgraph_query_performance = np.array(resolve_performance_variance_log(in_ds_kgraph_query_performance_log_path))
    kgraph_query_performance = np.array(resolve_performance_variance_log(kgraph_query_performance_log_path))
    query_performance = np.array(resolve_performance_variance_log(query_performance_log_path))
    query_performances = [query_performance]
    for i in range(9):
        query_performances.append(np.array(resolve_performance_variance_log(query_performance_log_paths[i])))
    query_performances = np.array(query_performances)
    
    query_performance_avg = np.sum(query_performances, axis=0) / len(query_performances)

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
    # KGraph = read_ivecs(kgraph_path)
    KGraph = read_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs_clean'))
    lengths = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'))
    # q_lengths = get_query_length(X, Q)
    n_rknn = read_ibin_simple(ind_path)
    # n_rknn2 = read_ibin_simple(ind_path2)
    # indegree = get_indegree_list(G)
    # write_ibin_simple(hnsw_ind_path, indegree)
    # indegree = read_ibin_simple(hnsw_ind_path)
    # revG = get_reversed_graph_list(KGraph)
    # save revG: a list of list
    # write_obj(reversed_kgraph_path, revG)
    revG = read_obj(reversed_kgraph_path)
    
    
    # hardness = []
    # for i in range(Q.shape[0]):
    #     if i % 100 == 0:
    #         print(f'{i} / {Q.shape[0]} ')
    #     query = Q[i]
    #     gt = GT[i][:50]
    #     knn_dist = GT_dist[i][49]
    #     flag = False
    #     unsettled = set(gt)
    #     for j in range(50, 10000):
    #         subset = GT[i][:j]
    #         update_reachable(KGraph, gt, subset, unsettled)
    #         if len(unsettled) == 0:
    #             flag = True
    #             hardness.append(j)
    #             break
    #     if flag == False:
    #         hardness.append(10000)
    #         print(i)
    # hardness = []
    # print(f'save to file {me_greedy_path_opt}')
    # with open(me_greedy_path_opt, 'w') as f:
    # for i in range(Q.shape[0]):
    #     if i % 100 == 0:
    #         print(f'{i} / {Q.shape[0]}: {datetime.now()} ')
    #     me_greedy = get_me_greedy_usg(KGraph, GT[i][:50], GT[i], revG)
    #     hardness.append(me_greedy)
    #     # f.write(f'{me_greedy},')
    # write_ibin_simple(me_greedy_path_opt, np.array(hardness))
    
    
    greedy_me = read_ibin_simple(me_greedy_path)
    print(f'greedy_me: {np.average(greedy_me)}')
        
    
