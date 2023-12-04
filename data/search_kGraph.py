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

def get_outlink_density_KGraph(n_rknn, G, KG, hubs_percent=[0.01,0.1,1,5,10]):
    hubs_percent = np.array(hubs_percent)
    nhub = (hubs_percent * n_rknn.shape[0] * 0.01).astype('int32')
    outlink_density = []
    for hubs_num in nhub:
        print(hubs_num)
        hubs = np.argsort(n_rknn)[-hubs_num:]
        sum_inside = 0
        sum_outside = 0
        for hub in hubs:
            neighbors = G[hub][:KG+1].copy()
            neighbors = np.delete(neighbors, np.where(neighbors == hub))
            intersect = np.intersect1d(neighbors, hubs)
            sum_inside += len(intersect)
            sum_outside += len(neighbors) - len(intersect)
        print(sum_inside, sum_outside, sum_inside + sum_outside, sum_inside / (sum_inside + sum_outside))
        outlink_density.append(sum_inside / (sum_inside + sum_outside))
    print(outlink_density)

def get_density_scatter(lengths, n_rknn):
    x = lengths
    y = n_rknn
    y = np.array([np.log2(i) if i > 0 else i for i in y])
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

    fig = plt.figure(figsize=(6,6))
    using_mpl_scatter_density(fig, x, y)
    plt.xlabel('Norm. distance to mean')
    plt.ylabel('$k$-occurence (log$_{2}$)')
    # plt.ylim(0, 200)
    # plt.tight_layout()
    plt.savefig(f'./figures/{dataset}-norm-k-occur{KG}.png')

def get_corr_distance2mean_avg_k_occur(lengths, n_rknn):
    min_len = np.min(lengths)
    max_len = np.max(lengths)
    nparts = 20
    len_range = np.linspace(min_len, max_len, nparts+1)
    print(len_range)
    k_occur = [[] for i in range(nparts+1)]
    for i in range(X.shape[0]):
        if i % 100000 == 0:
            print(i)
        # find the range of lengths[i] belongs to
        tag = False
        for j in range(nparts):
            if lengths[i] >= len_range[j] and lengths[i] < len_range[j+1]:
                k_occur[j].append(n_rknn[i])
                tag = True
                break
        if not tag:
            k_occur[-1].append(n_rknn[i])
            
    k_occurs = [np.array(l) for l in k_occur]
    k_occur_medians = [np.median(l) for l in k_occurs]
    print('median')
    print(k_occur_medians)
    print('average')
    k_occur_avg = [np.average(l) for l in k_occurs]
    print(k_occur_avg)
    fig = plt.figure(figsize=(6,6))
    plt.plot(len_range, k_occur_medians, **plot_info[0])
    plt.xlabel('Norm. distance to mean')
    plt.ylabel('Median of $k$-occurence')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'./figures/{dataset}-norm-median-k-occur{KG}.png')
    plt.close()
    fig = plt.figure(figsize=(6,6))
    plt.plot(len_range, k_occur_avg, **plot_info[0])
    plt.xlabel('Norm. distance to mean')
    plt.ylabel('Average of $k$-occurence')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'./figures/{dataset}-norm-average-k-occur{KG}.png')

def get_norm_dist_hists(lengths):
    plt.hist(lengths, bins=50, color='grey', edgecolor='black', linewidth=0.5)
    # plt.yscale('log')
    plt.xlabel('Distance to mean')
    plt.ylabel('Frequency ($10^4$)')
    plt.ylim(0, 250000)
    plt.yticks([0, 50000, 100000, 150000, 200000, 250000], ['0', '5', '10', '15', '20', '25'])
    # plt.yscale('log')
    # plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f'./figures/{dataset}-norm-hist.png')

def get_k_occurrence_hist(n_rknn):
    plt.hist(n_rknn, bins=50, edgecolor='black', linewidth=0.5, color='grey')
    # plt.yscale('log')
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500], ['0', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000', '1500'], fontsize=10)
    plt.xlabel('$k$-occurence')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'./figures/{dataset}-hist.png')

def get_corr_k_occurrence_visit_times_during_search(n_rknn):
    # filter the points with glanced_points > 200
    # filter_pos = np.where(glanced_points > 200)[0]
    # filter_r_knn = n_rknn[filter_pos]
    # print(np.min(filter_r_knn), np.percentile(filter_r_knn, 1), np.percentile(filter_r_knn, 5), np.percentile(filter_r_knn, 50))
    

    plt.plot(n_rknn, visited_points, 'o', color='black', markersize=0.1)
    plt.xlabel('indegree of kNN graph')
    plt.ylabel('visited_times')
    # plt.xlim(0, 1200)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title(f'{dataset} indegree correlation')
    plt.savefig(f'./figures/{dataset}-visit-hub-correlation.png')
    
    write_fbin_simple(f'./figures/{dataset}-kgraph-visit-hub-correlation.fbin', glanced_points)
    
    
    print(np.corrcoef(n_rknn, glanced_points))
    plt.close()
    plt.plot(n_rknn, glanced_points, 'o', color='black', markersize=0.1)
    plt.xlabel('Indegree of points')
    plt.ylabel('Visted times')
    # plt.xlim(0, 1200)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title(f'{dataset} indegree correlation')
    plt.tight_layout()
    plt.savefig(f'./figures/{dataset}-glance-hub-correlation.png')

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

def search_with_ep(graph, vec, query, k, KG, efsearch, eps, ndc=0, nhops=0, visited_points=None, glanced_points=None):
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
        visited_points[top.id] += 1
        for nei in graph[top.id][:KG+1]:
            if nei == top.id or nei in visited:
                continue
            visited.add(nei)
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
        
def search(graph, vec, query, k, KG, efsearch, visited_points, glanced_points):
    eps = select_ep(vec.shape[0])
    ndc = 0
    nhops = 0
    return search_with_ep(graph, vec, query, k, KG, efsearch, eps, ndc, nhops, visited_points, glanced_points)
    
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
 
def do_expr(X, Q, GT, k, KG, n_rknn, lengths):
    
    # efss = [100, 200, 300, 400, 500]
    efss = [3000, 4000, 5000, 5200, 5400, 5600, 5800, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000, 12000, 15000, 18000, 20000, 25000, 30000, 35000, 40000]
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
            topk, ndc, nhops = search(G, X, q, k, KG, efs, visited_points, glanced_points)
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
        print(f'\ttp_lengths: {sum_tp_lengths / sum_tps}, fp_lengths: {sum_fp_lengths / sum_fps}, fn_lengths: {sum_fn_lengths / sum_fns}, tp-fn: {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns}')
        print(f'{sum_tp_lengths / sum_tps:.2f} & {sum_fp_lengths / sum_fps:.2f} & {sum_fn_lengths / sum_fns:.2f} & {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns:.2f}')
        write_fbin_simple(f'./figures/{dataset}-kgraph-visit-hub-correlation-ef{efs}.fbin', visited_points)
        write_fbin_simple(f'./figures/{dataset}-kgraph-glance-hub-correlation-ef{efs}.fbin', glanced_points)
    
    
    
source = './data/'
dataset = 'deep'
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    graph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth_500.ivecs')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs')
    KG = 100
    ind_path = os.path.join(source, dataset, f'{dataset}_ind_{KG}.ibin')

    X = read_fvecs(base_path)
    G = read_ivecs(graph_path)
    Q = read_fvecs(query_path)
    GT = read_ivecs(GT_path)
    n_rknn = get_reversed_knn_number(G, KG)
    write_ibin_simple(ind_path, n_rknn)
    n_rknn = read_ibin_simple(ind_path)
    # print(get_skewness(n_rknn))
    
    # print(get_bidirectional_neighbors(G, KG))
    
    lengths = compute_lengths(X)
    # write_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'), lengths)
    lengths = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'))
    
    # get norm length histogram
    # get_norm_dist_hists(lengths)
    
    # plot the histogram of n_rknn
    get_k_occurrence_hist(n_rknn)
    
    # get density scatter figure
    # get_density_scatter(lengths, n_rknn)
    
    # get correlation between distance to mean and average k-occurence
    # get_corr_distance2mean_avg_k_occur(lengths, n_rknn)
        
    # insertion_order = np.arange(X.shape[0])[::-1]
    # print(np.corrcoef(n_rknn, insertion_order))
    
    # out-link density
    # get_outlink_density_KGraph(n_rknn, G, KG)
    
    
    # assert KG < G.shape[1]
        
    k = 50
    do_expr(X, Q, GT, k, KG, n_rknn, lengths)
    
    # get_corr_k_occurrence_visit_times_during_search(n_rknn)
