import queue
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

    fig = plt.figure(figsize=(12,10))
    using_mpl_scatter_density(fig, x, y)
    # plt.xlabel('k-occurrence (450)')
    # plt.xlabel('Query difficulty')
    # plt.xlabel('reach_delta_0_rigorous')
    # plt.xlabel('metric for dynamic bound')
    # plt.xlabel('delta_0')
    # plt.xlabel(r'$\delta_0^{\forall}-g@0.98$')
    # plt.xlabel(r'$ME^{\forall}_{\delta_0}$@0.98-exhausted')
    plt.xlabel(r'$Kgraph-400-\hat{ME^{0.96}_{\delta_0}}$@0.98-exhausted')
    # plt.xlabel(r'$ME^{\forall}_{\delta_0}-reach$@0.98')
    # plt.ylabel(r'NDC (recall@50>0.98)')
    plt.ylabel('HNSW NDC')
    # plt.xlabel('local intrinsic dimensionality')
    # plt.ylabel('1NN distance')
    # plt.tight_layout()
    plt.xlim(0, 40000)
    plt.ylim(0, 20000)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k_occurs-lid-scatter.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-k_occurs-lid-scatter.png')
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k-occur-1NN-dist.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-k-occur-1NN-dist.png')
    plt.savefig(f'./figures/{dataset}/{dataset}-query-difficulty.png')
    print(f'save to figure ./figures/{dataset}/{dataset}-query-difficulty.png')

def resolve_performance_variance_log(file_path):
    print(f'read {file_path}')
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
    
def resolve_performance_variance_log_multi(file_path):
    print(f'read {file_path}')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        ndcs = []
        for line in lines:
            if(len(line) < 30):
                continue
            line = line.strip()
            ndc = line[:-1].split(',')
            ndc = [int(x) for x in ndc]
            ndcs.append(ndc)
        return ndcs

def resolve_delta_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        deltas_list = []
        for line in lines:
            if(len(line) < 30):
                continue
            line = line.strip()
            deltas = line[:-1].split(',')
            deltas = [float(x) for x in deltas[2:]]
            deltas_list.append(deltas)
        return deltas_list

source = './data/'
result_source = './results/'
dataset = 'deep'
idx_postfix = '_plain'
efConstruction = 500
Kbuild = 499
M = 16
R = 32
L = 40
C = 500
target_recall = 0.98
target_prob = 0.96
select = 'kgraph'
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
    standard_hnsw_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}_hnsw.ibin{idx_postfix}')
    reversed_kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs_reversed')
    result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_.log{idx_postfix}_unidepth-level')
    ma_dist_path = os.path.join(source, dataset, f'{dataset}_ma_distance.fbin')
    ma_base_dist_path = os.path.join(source, dataset, f'{dataset}_ma_base_distance.fbin')
    me_greedy_path = os.path.join(source, dataset, f'{dataset}_me_greedy.ibin')
    me_greedy_path_opt = os.path.join(source, dataset, f'{dataset}_me_greedy.ibin_usg')
    delta_result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_delta.log')
    delta0_point_path = os.path.join(source, dataset, f'{dataset}_delta0_point_K{Kbuild}.ibin')
    self_delta0_max_knn_rscc_point_recall_path = os.path.join(source, dataset, f'{dataset}_self_delta0_max_knn_rscc_point_recall{target_recall}_K{Kbuild}.ibin')
    delta0_rigorous_point_path = os.path.join(source, dataset, f'{dataset}_delta0_rigorous_point_K{Kbuild}.ibin')
    delta0_max_knn_rscc_point_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_poin_K{Kbuild}.ibin')
    kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log')
    in_ds_kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log_in-dataset')
    in_ds_kgraph_query_performance_recall_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance{target_recall}.log_in-dataset')
    query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance{target_recall}.log_plain')
    query_performance_log_paths = []
    for i in range(3, 11):
        query_performance_log_paths.append(os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance{target_recall}.log_plain_shuf{i}'))

    query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
                                                   '_K400.ibin_clean'))
    query_performance = np.array(resolve_performance_variance_log(query_performance_log_path))
    query_performances = [query_performance]
    for i in range(4):
        query_performances.append(np.array(resolve_performance_variance_log(query_performance_log_paths[i])))
    query_performances = np.array(query_performances)
    query_performance = np.average(query_performances, axis=0)        
        
    get_density_scatter(query_hardness, query_performance)
    
    # hardness_K = 150
    # hardness_path = os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}_K{hardness_K}.ibin_clean')
    # hardness = read_ibin_simple(hardness_path)
    
    
    # query_performance_avg = np.sum(query_performances, axis=0) / len(query_performances)
    
    # in_ds_kgraph_query_performance = np.array(resolve_performance_variance_log(in_ds_kgraph_query_performance_log_path))
    # in_ds_kgraph_query_performance_recall = np.array(resolve_performance_variance_log(in_ds_kgraph_query_performance_recall_log_path))[:50000]

    # recalls = []
    # with open(result_path, 'r') as f:
    #     lines = f.readlines()
    #     for i in range(3, len(lines), 7):
    #         line = lines[i].strip().split(',')[:-1]
    #         recalls.append(np.array([int(x) for x in line]))

    # low_recall_positions = []
    # for recall in recalls:
    #     low_recall_positions.append(np.where(recall < 40)[0])
    # X = read_fvecs(base_path)
    # # G = read_hnsw_index_aligned(index_path, X.shape[1])
    # # G = read_hnsw_index_unaligned(index_path, X.shape[1])
    # Q = read_fvecs(query_path)
    # # Q = read_fvecs(query_path)
    # GT = read_ivecs(GT_path)
    # GT_dist = read_fvecs(GT_dist_path)
    # KGraph = read_ivecs(kgraph_path)

    # q_lids = get_lids(GT_dist[:, :50], 50)

    
    # KGraph = read_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs_clean'))
    # lengths = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'))
    # q_lengths = get_query_length(X, Q)
    # n_rknn = read_ibin_simple(ind_path)
    # n_rknn2 = read_ibin_simple(ind_path2)
    # indegree = get_indegree_list(G)
    # write_ibin_simple(hnsw_ind_path, indegree)
    # indegree = read_ibin_simple(hnsw_ind_path)
    # revG = get_reversed_graph_list(KGraph)
    # save revG: a list of list
    # write_obj(reversed_kgraph_path, revG)
    # revG = read_obj(reversed_kgraph_path)
    
    
    # me_greedy = read_ibin_simple(self_delta0_max_knn_rscc_point_recall_path)[:50000]
    # me_greedy = read_ibin_simple(delta0_max_knn_rscc_point_recall_path)
    # delta0_point = read_ibin_simple(delta0_point_path)
    
    
    # me_delta0 = []
    # for i in range(delta0_point.shape[0]):
    #     if i % 1000 == 0:
    #         print(i)
    #     # if i != 8464:
    #     #     continue
    #     me = get_me(G, GT[i], delta0_point[i], 50, target_recall)
    #     me_delta0.append(me)
    # write_ibin_simple(me_delta0_path, np.array(me_delta0))
    # me_delta0 = read_ibin_simple(me_delta0_path)    
    # reach_delta_0 = []
    # for i in range(me_greedy.shape[0]):
    #     if i % 1000 == 0:
    #         print(i)
    #     # if i != 1106:
    #     #     continue
    #     reach_delta_0.append(len(get_reachable_of_a_group(KGraph, GT[i][:50], GT[i][:me_greedy[i]])))
        
    # write_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0_max_knn_rscc_recall{target_recall}.ibin'), np.array(reach_delta_0))
    # reach_delta_0 = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0_max_knn_rscc_recall{target_recall}.ibin'))
    
    # plt.hist(me_delta0, bins=50, edgecolor='black')
    # plt.xlabel(r'$\hat{ME^{\forall}_{\delta_0}}$@0.98')
    # plt.ylabel('#queries (log)')
    # plt.yscale('log')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-me_delta_0.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-me_delta_0.png')
    
    # # write_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0_rigorous.ibin'), np.array(reach_delta_0))
    # # reach_delta_0 = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0_rigorous.ibin'))

    # write_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0_max_knn_rscc.ibin'), np.array(reach_delta_0))
    # reach_delta_0 = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0_max_knn_rscc.ibin'))
        
    # write_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0.ibin'), np.array(reach_delta_0))
    # reach_delta_0 = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0.ibin'))
    # plt.hist(reach_delta_0, bins=100, edgecolor='black')
    # plt.xlabel('reach_delta_0')
    # plt.ylabel('number of points')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-reach_delta_0.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-reach_delta_0.png')
    # me_greedy_naive = read_ibin_simple(me_greedy_path)
    
    # delta0 = [ get_delta_0_from_nn_point(GT_dist[i], 50, delta0_point[i]) for i in range(delta0_point.shape[0])]
    
    # replace all zeros with 0.01
    # delta_0 = np.array([0.005 if x == 0 else x for x in delta_0])
    # plt.hist(delta_0, bins=50, edgecolor='black')
    # plt.xlabel('delta_0')
    # plt.ylabel('number of points')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-delta_0.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-delta_0.png')
    # plt.close()
    
    # sorted_reach_delta_0 = np.sort(reach_delta_0)
    
    # metric = delta_0 * np.log(reach_delta_0 / 50)
    # replace all zeros with 0.01
    # metric = np.array([0.005 if x == 0 else x for x in metric])
    # plt.hist(metric, bins=50, edgecolor='black')
    # plt.xlabel('dynamic bound')
    # plt.ylabel('number of points')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-reach_delta_0-delta_0.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-reach_delta_0-delta_0.png')
    # plt.close()
    
    # deltas = resolve_delta_log(delta_result_path)
    # deltas = deltas[-Q.shape[0]:]
    # hops_to_break_delta0 = []
    # for i in range(len(deltas)):
    #     # find the first element that is smaller than delta_0 in deltas[i]
    #     pos = np.where(deltas[i] < delta_0[i])[0]
    #     hops_to_break_delta0.append(pos[0] if len(pos) > 0 else len(deltas[i]))
    
    # get_density_scatter(q_lids, query_performance)
    
    # plt.hist(hops_to_break_delta0, bins=50, edgecolor='black')
    # plt.xlabel('hops to break delta_0')
    # plt.ylabel('number of points')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-hops_to_break_delta_0.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-hops_to_break_delta_0.png')
    
    # pos = np.where(kgraph_query_performance < me_greedy)
    # for p in pos[0]:
    #     print(f'{p}: ndc: {kgraph_query_performance[p]}, me: {me_greedy[p]}')
    
    
    # get_density_scatter(pagerank, in_ds_kgraph_query_performance)
    
    # special_q = []
    # for i in range(Q.shape[0]):
    #     if kgraph_query_performance_recall[i] < 3000 and me_delta0[i] > 200:
    #         special_q.append(i)
    # # sort the special_q by reach_delta_0
    # special_q = np.array(special_q)
    # special_q = special_q[np.argsort(kgraph_query_performance_recall[special_q])]
    # for q in special_q:
    #     print(f'{q}: metric:{metric[q]}, d1:{GT_dist[q][0]}, d50:{GT_dist[q][49]}, me_delta_0: {me_delta0[q]}, delta_0: {delta_0[q]}, delta_0_point: {delta0_point[q]}, ndc: {kgraph_query_performance_recall[q]}')
    

    # special_q = []
    # for i in range(50000):
    #     if in_ds_kgraph_query_performance_recall[i] > 20000:
    #         special_q.append(i)
    # # sort the special_q by reach_delta_0
    # special_q = np.array(special_q)
    # in_ds_kgraph_query_performance_recall = np.array(in_ds_kgraph_query_performance_recall)
    # special_q = special_q[np.argsort(in_ds_kgraph_query_performance_recall[special_q])]
    # for q in special_q:
    #     print(f'{q}: delta_0_point: {me_greedy[q]}, ndc: {in_ds_kgraph_query_performance_recall[q]}')

