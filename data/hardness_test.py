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

def get_query_k_occur(GT, n_rknn):
    q_k_occurs = []
    for i in range(GT.shape[0]):
        gt = GT[i][:50]
        sum_k_occur = 0
        for j in range(50):
            sum_k_occur += n_rknn[gt[j]]
        q_k_occurs.append(sum_k_occur / 50)
    return np.array(q_k_occurs)
    
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
    # plt.xlabel(r'$Kgraph-100-\hat{ME^{0.96}_{\delta_0}}$@0.98-exhausted')
    # plt.xlabel(r'$ME^{\forall}_{\delta_0}-reach$@0.98')
    # plt.ylabel(r'NDC (recall@50>0.98)')
    # plt.ylabel('HNSW NDC')
    # plt.xlabel('local intrinsic dimensionality')
    # plt.ylabel('1NN distance')
    # plt.tight_layout()
    plt.xlim(0, 100000)
    plt.ylim(0, 100000)
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

source = './data/'
result_source = './results/'
dataset = 'rand100'
idx_postfix = '_plain'
efConstruction = 500
Kbuild = 500
M = 64
R = 32
L = 40
C = 500
target_recall = 0.9
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
                                                   '_K100.ibin_clean'))
    
    
    # K = 100
    # ind_path = os.path.join(source, dataset, f'{dataset}_ind_{K}.ibin')
    # G = read_ivecs(kgraph_path)
    # n_rknn = get_reversed_knn_number(G, K)
    # write_ibin_simple(ind_path, n_rknn)
    # n_rknn = read_ibin_simple(ind_path)
    # GT = read_ivecs(GT_path)
    # k_occur = get_query_k_occur(GT, n_rknn)
    
    
    
    query_performances = []
    query_performances.append(np.array(resolve_performance_variance_log(query_performance_log_path)))
    for i in range(1):
        query_performances.append(np.array(resolve_performance_variance_log(query_performance_log_paths[i])))
    query_performances = np.array(query_performances)
    # query_performance = np.median(query_performances, axis=0)        
    kgraph_query_performance_recall_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance{target_recall}.log_clean')
    kgraph_query_performance_recall = np.array(resolve_performance_variance_log(kgraph_query_performance_recall_log_path))
    
    # get_density_scatter(query_hardness, query_performances[0])
    get_density_scatter(query_performances[0], query_performances[1])
    
    
    

    # get_density_scatter(query_hardness, query_performance)
