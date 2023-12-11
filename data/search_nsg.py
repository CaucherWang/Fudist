from utils import *
from queue import PriorityQueue
import os

fig = plt.figure(figsize=(6,4.8))

def read_nsg(filename):
    data = np.memmap(filename, dtype='uint32', mode='r')
    width = int(data[0])
    ep = int(data[1])
    print(f'width: {width}, ep: {ep}')
    data_len = len(data)
    edge_num = 0
    cur = 2
    graphs = []
    max_edge = 0
    while cur < data_len:
        if len(graphs) % 100000 == 0:
            print(len(graphs))
        edge_num += data[cur]
        max_edge = max(max_edge, data[cur])
        tmp = []
        for i in range(data[cur]):
            tmp.append(data[cur + i + 1])
        cur += data[cur] + 1
        graphs.append(tmp)
    print(f'edge number = {edge_num}')
    print(f'node number = {len(graphs)}')
    print(f'max degree = {max_edge}')
    return ep, graphs

class node:
    def __init__(self, id, dist):
        self.id = id
        self.dist = dist
    
    def __lt__(self, other):
        # further points are prior
        return self.dist > other.dist
    
    def __str__(self) -> str:
        return f'id: {self.id}, dist: {self.dist}'

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
        # if visited_points is not None:
        #     visited_points[top.id] += 1
        for nei in graph[top.id]:
            if nei in visited:
                continue
            visited.add(nei)
            # if glanced_points is not None:
            #     glanced_points[nei] += 1
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
    gts = set(groundtruth[: len(topk)])
    for id in ids:
        if id in groundtruth:
            cnt += 1
    return cnt 

def stat_out_degree(graph):
    out_degree = np.zeros(len(graph), dtype=np.int32)
    for i in range(len(graph)):
        out_degree[i] = len(graph[i])
    # percentiles
    print(np.percentile(out_degree, [50,90,95,99, 99.5, 99.95]))
    # max
    print(np.max(out_degree))
    # average
    print(np.average(out_degree))
    
def do_expr(X, Q, GT, k, eps, n_rknn, lengths, indegree):
    
    # efss = [100, 200, 300, 400, 500]
    # efss = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 4000, 5000, 5200, 5400, 5600, 5800, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000, 12000, 15000, 18000, 20000, 25000, 30000, 35000, 40000]
    efss = [13000, 14000, 14500]
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
            topk, ndc, nhops = search(G, X, q, k,  efs, visited_points, glanced_points, eps=eps)
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
            if idx % 1000 == 0:
                print(idx)
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
        write_fbin_simple(f'./figures/{dataset}-nsg-visit-hub-correlation-ef{efs}.fbin', visited_points)
        write_fbin_simple(f'./figures/{dataset}-nsg-glance-hub-correlation-ef{efs}.fbin', glanced_points)
 

source = './data/'
dataset = 'rand100'
R = 100
L = 200
C = 500
Kbuild = 16
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    graph_path = os.path.join(source, dataset, f'{dataset}_L{L}_R{R}_C{C}.nsg')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs')
    KG = 32
    ind_path = os.path.join(source, dataset, f'{dataset}_ind_{KG}.ibin')
    Kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')
    nsg_ind_path = os.path.join(source, dataset, f'{dataset}_nsg_L{L}_R{R}_C{C}_ind_{KG}.ibin')

    X = read_fvecs(base_path)
    ep, G = read_nsg(graph_path)
    Q = read_fvecs(query_path)[:1000]
    # Q = read_fvecs(query_path)
    GT = read_ivecs(GT_path)
    # K_Graph = read_ivecs(Kgraph_path)
    # lengths = compute_lengths(X)
    # write_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'), lengths)
    # lengths = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'))
    # # n_rknn = get_reversed_knn_number(G, KG)
    # # write_ibin_simple(ind_path, n_rknn)
    # n_rknn = read_ibin_simple(ind_path)
    # # stat_out_degree(G)
    # # print(get_graph_quality_list(G, K_Graph, KG))
    
    # # indegree = get_indegree_list(G)
    # # write_ibin_simple(nsg_ind_path, indegree)
    # indegree = read_ibin_simple(nsg_ind_path)
    # get_indegree_hist(indegree, dataset, 'nsg')
    # # find the elements with the largest indegree
    # get_outlink_density(indegree, G, X.shape[0])    

    # print(np.percentile(indegree, [10,25,50,75,90,95,99]))
    # plt.hist(indegree, bins=100)
    # plt.savefig(f'./figures/{dataset}-nsw-indegree-hist.png')
        
    
    # insertion_order = np.arange(X.shape[0])[::-1]
    # print(np.corrcoef(insertion_order, indegree))
    # print(np.corrcoef(n_rknn, indegree))
    # data = np.vstack((insertion_order, n_rknn, indegree)).T    
    # import pandas
    # import statsmodels.api as sm
    # from statsmodels.formula.api import ols
    # data = pandas.DataFrame(data, columns=['insertion_order', 'n_rknn', 'indegree'])
    # model = ols("indegree ~ n_rknn + insertion_order", data).fit()
    # print(model.summary())

    # print(get_skewness(indegree))
    # print(np.corrcoef(np.arange(1000000), indegree))
    # print(np.corrcoef(n_rknn, indegree))
    # # plot the scatter plot of n_rknn and indegree
    # plt.plot(n_rknn, indegree, 'o', color='black', markersize=0.5)
    # plt.xlabel('indegree of kNN graph')
    # plt.ylabel('indegree of NSW')
    # plt.xlim(0, 1200)
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title(f'{dataset} indegree correlation')
    # plt.savefig(f'./figures/{dataset}-nsw-indegree-correlation.png')

    
        
    k = 50
    # do_expr(X, Q, GT, k, [ep], n_rknn, lengths, indegree)
    # visited_points = np.zeros(X.shape[0], dtype=np.int32)
    # glanced_points = np.zeros(X.shape[0], dtype=np.int32)
    
    efss = [20025]
    # efss = [1500,2000,2500,3000]
    for efs in efss:
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
        recall = 0
        # print()
        for q in Q:
            topk, ndc, nhops = search(G, X, q, k, efs, eps=[ep])
            gt = GT[idx][:k]
            recall = get_recall(topk, gt)
            
            # tps, fps, fns = decompose_topk_to_3sets(np.array(topk), np.array(gt))
            # sum_tps += len(tps)
            # sum_fps += len(fps)
            # sum_fns += len(fns)
            
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
            
            # idx += 1
            # if(idx % 1000 == 0):
            #     print(idx)
            # sum_ndc += ndc
            # sum_nhops += nhops
            # sum_recall += recall
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
    
    # # print(np.corrcoef(visited_points, indegree))
    # visit_indegree = np.concatenate((visited_points.reshape(-1, 1), indegree.reshape(-1, 1)), axis=1)
    # # filter the rows with 0 visit
    # sotred_indices = np.argsort(visit_indegree[:, 0])[-200:]
    # sorted_indegree = visit_indegree[sotred_indices][1:]
    # print(np.corrcoef(sorted_indegree[:, 0], sorted_indegree[:, 1]))
    # # sorted_visited_points = sorted(visited_points)
    # # largest_index = [list(visited_points).index(sorted_visited_points[-i]) for i in range(2, 10000)]
    # # non_zero_pos = np.where(visited_points > 0)
    # # cleaned_visited_points = visited_points[non_zero_pos]
    # # cleaned_indegree = indegree[non_zero_pos]
    # # cleaned_visited_points = visited_points[largest_index]
    # # cleaned_indegree = indegree[largest_index]
    # # print(np.corrcoef(cleaned_visited_points, cleaned_indegree))
    # # plot the scatter plot of visited times and indegree
    # import matplotlib.pyplot as plt
    # plt.plot(sorted_indegree[0], sorted_indegree[1], 'o', color='black', markersize=1)
    # plt.xlabel('visited times')
    # plt.ylabel('indegree of NSW')
    # # plt.xlim(0, 1200)
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title(f'{dataset} visited indegree correlation')
    # plt.savefig(f'./figures/{dataset}-nsw-visited-indegree-correlation.png')

    # write_ibin_simple(f'./figures/{dataset}-nsw-visit-hub-correlation.ibin', visited_points)
    # write_ibin_simple(f'./figures/{dataset}-nsw-glance-hub-correlation.ibin', glanced_points)
    # glanced_points = read_ibin_simple(f'./figures/{dataset}-nsw-glance-hub-correlation.ibin')
    # remove the max
    # max_pos = np.where(glanced_points > 800)
    # glanced_points = np.delete(glanced_points, max_pos)
    # indegree = np.delete(indegree, max_pos)
    

    # plt.plot(indegree, visited_points, 'o', color='black', markersize=0.1)
    # plt.xlabel('indegree of NSW')
    # plt.ylabel('visited_times')
    # # plt.xlim(0, 1200)
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.title(f'{dataset} indegree correlation')
    # plt.savefig(f'./figures/{dataset}-nsw-visit-hub-correlation.png')
    
    # print(np.corrcoef(n_rknn, glanced_points))
    # plt.close()
    # plt.plot(indegree, glanced_points, 'o', color='black', markersize=0.1)
    # plt.xlabel('Indegree of points')
    # plt.ylabel('Visted times')
    # # plt.xlim(0, 1200)
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.title(f'{dataset} indegree correlation')
    # plt.savefig(f'./figures/{dataset}-nsw-glance-hub-correlation.png')
    
    # # filter the points with glanced_points > 200
    # filter_pos = np.where(glanced_points > 200)[0]
    # filter_r_knn = n_rknn[filter_pos]
    # print(np.min(filter_r_knn), np.percentile(filter_r_knn, 1), np.percentile(filter_r_knn, 5), np.percentile(filter_r_knn, 50))
