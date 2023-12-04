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
dataset = 'gauss200'
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    graph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs')
    KG = 32
    ind_path = os.path.join(source, dataset, f'{dataset}_ind_{KG}.ibin')

    X = read_fvecs(base_path)
    G = read_ivecs(graph_path)
    Q = read_fvecs(query_path)[:1000]
    GT = read_ivecs(GT_path)
    # n_rknn = get_reversed_knn_number(G, KG)
    # write_ibin_simple(ind_path, n_rknn)
    n_rknn = read_ibin_simple(ind_path)
    # print(get_skewness(n_rknn))
    
    # print(get_bidirectional_neighbors(G, KG))
    
    # lengths = compute_lengths(X)
    # write_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'), lengths)
    lengths = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'))
    
    # x = lengths
    # y = n_rknn
    # # y = np.array([np.log2(i) if i > 0 else i for i in y])
    # # y = np.log(n_rknn)
    # print(np.max(x), np.max(y))
    # fig = plt.figure(figsize=(7,6))

    # white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    #     (0, '#ffffff'),
    #     (1e-20, '#440053'),
    #     (0.2, '#404388'),
    #     (0.4, '#2a788e'),
    #     (0.6, '#21a784'),
    #     (0.8, '#78d151'),
    #     (1, '#fde624'),
    # ], N=256)

    # def using_mpl_scatter_density(fig, x, y):
    #     ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    #     density = ax.scatter_density(x, y, cmap=white_viridis)
    #     fig.colorbar(density, label='#points per pixel')

    # using_mpl_scatter_density(fig, x, y)
    # plt.xlabel('Norm. distance to mean')
    # plt.ylabel('$k$-occurence')
    # plt.ylim(0, 600)
    # # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}-norm-k-occur{KG}.png')
    
    # min_len = np.min(lengths)
    # max_len = np.max(lengths)
    # nparts = 20
    # len_range = np.linspace(min_len, max_len, nparts+1)
    # print(len_range)
    # k_occur = [[] for i in range(nparts+1)]
    # for i in range(X.shape[0]):
    #     if i % 100000 == 0:
    #         print(i)
    #     # find the range of lengths[i] belongs to
    #     tag = False
    #     for j in range(nparts):
    #         if lengths[i] >= len_range[j] and lengths[i] < len_range[j+1]:
    #             k_occur[j].append(n_rknn[i])
    #             tag = True
    #             break
    #     if not tag:
    #         k_occur[-1].append(n_rknn[i])
            
    # k_occurs = [np.array(l) for l in k_occur]
    # k_occur_medians = [np.median(l) for l in k_occurs]
    # print('median')
    # print(k_occur_medians)
    # print('average')
    # k_occur_avg = [np.average(l) for l in k_occurs]
    # print(k_occur_avg)
    # fig = plt.figure(figsize=(6,6))
    # plt.plot(len_range, k_occur_medians, **plot_info[0])
    # plt.xlabel('Norm. distance to mean')
    # plt.ylabel('Median of $k$-occurence')
    # plt.yscale('log')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}-norm-median-k-occur{KG}.png')
    # plt.close()
    # fig = plt.figure(figsize=(6,6))
    # plt.plot(len_range, k_occur_avg, **plot_info[0])
    # plt.xlabel('Norm. distance to mean')
    # plt.ylabel('Average of $k$-occurence')
    # plt.yscale('log')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}-norm-average-k-occur{KG}.png')
    
    
            
    
    # plt.hist(lengths, bins=50, color='grey', edgecolor='black', linewidth=0.5)
    # # plt.yscale('log')
    # plt.xlabel('Distance to mean')
    # plt.ylabel('Frequency ($10^4$)')
    # plt.ylim(0, 250000)
    # plt.yticks([0, 50000, 100000, 150000, 200000, 250000], ['0', '5', '10', '15', '20', '25'])
    # # plt.yscale('log')
    # # plt.xlim(0, 1)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}-norm-hist.png')

    
    # dist = compute_pairwiese_distance_simple(X, mean)
    # write_fbin_simple(os.path.join(source, dataset, f'{dataset}_mean_dist.fbin'), dist)
    # dist = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_mean_dist.fbin'))
    
    # normX = L2_norm_dataset(X)
    # write_fbin(os.path.join(source, dataset, f'{dataset}_norm_base.fbin'), normX)
    # # normX = read_fbin(os.path.join(source, dataset, f'{dataset}_norm_base.fbin'))
    # mean = np.mean(normX, axis=0)
    # print(mean)
    # dist = compute_pairwiese_distance_simple(normX, mean)
    # write_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_mean_dist.fbin'), dist)
    # dist = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_mean_dist.fbin'))
    
    # plot the histogram of n_rknn

    # plt.hist(n_rknn, bins=50, edgecolor='black', linewidth=0.5, color='grey')
    # plt.yscale('log')
    # plt.xlabel('$k$-occurence')
    # plt.ylabel('Frequency')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}-hist.png')

    # plot the histogram of n_rknn
    # plt.hist(dist, bins=200, edgecolor='black', linewidth=0.5, density=True)
    # # plt.yscale('log')
    # plt.xlabel('Distance to mean')
    # plt.ylabel('Frequency')
    # plt.savefig(f'./figures/{dataset}-norm-dist2mean.png')
    
    # plot the scatter point
    # print(np.corrcoef(n_rknn, dist))
    # rho, pval = spearmanr(n_rknn, dist)
    # print(rho, pval)
    # plt.plot(n_rknn, dist, 'o', color='black', markersize=0.1)
    # plt.xlabel('In-degree of kNN graph')
    # plt.ylabel('Distance to mean')
    # # plt.xlim(0, 1200)
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.title(f'{dataset} indegree correlation')
    # plt.savefig(f'./figures/{dataset}-corr-norm-dist2mean-k-occur{KG}.png')

    # print(np.corrcoef(n_rknn, lengths))
    # rho, pval = spearmanr(n_rknn, lengths)
    # print(rho, pval)
    # plt.plot(n_rknn, lengths, 'o', color='black', markersize=0.1)
    # plt.xlabel('In-degree of kNN graph')
    # plt.ylabel('Normalizzed distance to mean')
    # # plt.xlim(0, 1200)
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.title(f'{dataset} indegree correlation')
    # plt.savefig(f'./figures/{dataset}-corr-norm-dist2mean-k-occur{KG}.png')
    
    
    # insertion_order = np.arange(X.shape[0])[::-1]
    # print(np.corrcoef(n_rknn, insertion_order))
    
    # out-link density
    # hubs_percent = np.array([0.01,0.1,1,5,10])
    # nhub = (hubs_percent * n_rknn.shape[0] * 0.01).astype('int32')
    # outlink_density = []
    # for hubs_num in nhub:
    #     print(hubs_num)
    #     hubs = np.argsort(n_rknn)[-hubs_num:]
    #     sum_inside = 0
    #     sum_outside = 0
    #     for hub in hubs:
    #         neighbors = G[hub][:KG+1].copy()
    #         neighbors = np.delete(neighbors, np.where(neighbors == hub))
    #         intersect = np.intersect1d(neighbors, hubs)
    #         sum_inside += len(intersect)
    #         sum_outside += len(neighbors) - len(intersect)
    #     print(sum_inside, sum_outside, sum_inside + sum_outside, sum_inside / (sum_inside + sum_outside))
    #     outlink_density.append(sum_inside / (sum_inside + sum_outside))
    # print(outlink_density)

    
    
    # assert KG < G.shape[1]
        
    k = 50
    # visited_points = np.zeros(X.shape[0], dtype=np.int32)
    # glanced_points = np.zeros(X.shape[0], dtype=np.int32)
    
    
