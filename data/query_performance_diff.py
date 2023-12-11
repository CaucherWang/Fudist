from utils import *
from queue import PriorityQueue
import os
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
    # plt.xlabel(r'$LQC^{0.98}_{50}$ over KGraph')
    plt.xlabel(r'$LQC^{0.86}_{50}$ over 5 NSG instance')
    plt.ylabel(r'$LQC^{0.86}_{50}$ over 5 NSG instance')
    # plt.xlabel(r'$LQC^{0.98}_{50}$ over 10 HNSW instance')
    # plt.ylabel(r'$LQC^{0.98}_{50}$ over 10 HNSW instance')

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
    plt.xlim(0, 100000)
    plt.ylim(0, 100000)
    
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k_occurs-lid-scatter.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-k_occurs-lid-scatter.png')
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k-occur-1NN-dist.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-k-occur-1NN-dist.png')
    plt.savefig(f'./figures/{dataset}/{dataset}-query-performance-diff.png')
    print(f'save to figure ./figures/{dataset}/{dataset}-query-performance-diff.png')

    
def resolve_performance_variance_log(file_path):
    print(f'reading {file_path}')
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

params = {
    'gauss100':{'M': 50, 'ef': 500, 'L': 200, 'R': 100, 'C': 500},
    'rand100': {'M': 50, 'ef': 500, 'L': 200, 'R': 100, 'C': 500},
    'deep': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500},
    'sift': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500},
}
source = './data/'
result_source = './results/'
dataset = 'gauss100'
recall = 0.86
idx_postfix = '_plain'
efConstruction = params[dataset]['ef']
Kbuild = 100
M=params[dataset]['M']
L = params[dataset]['L']
R = params[dataset]['R']
C = params[dataset]['C']
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
    query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance{recall:.2f}.log_plain')
    # kgraph_query_performance_log_path2 = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K100_perform_variance.log')
    in_ds_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance.log_in-dataset_plain_1th')
    nsg_query_performance_log_path = os.path.join(result_source, dataset, f'{dataset}_nsg_L{L}_R{R}_C{C}_perform_variance{recall:.2f}.log')
    query_performance_log_paths = []
    nsg_query_performance_log_paths = []
    for i in range(3, 24):
        query_performance_log_paths.append(os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance{recall:.2f}.log_plain_shuf{i}'))
    for i in range(3, 12):
        nsg_query_performance_log_paths.append(os.path.join(result_source, dataset, f'{dataset}_nsg_L{L}_R{R}_C{C}_perform_variance{recall:.2f}.log_shuf{i}'))


    query_performance = np.array(resolve_performance_variance_log(query_performance_log_path))
    query_performances = [query_performance]
    for i in range(9):
        query_performances.append(np.array(resolve_performance_variance_log(query_performance_log_paths[i])))
        print(query_performances[-1].shape, np.average(query_performances[-1]))
    query_performances = np.array(query_performances)
    query_performance_avg = np.average(query_performances, axis=0)
    # half = int(query_performances.shape[0] / 2)
    # query_performance1 = np.average(query_performances[:half], axis=0)
    # query_performance2 = np.average(query_performances[half:], axis=0)  
    # get_density_scatter(query_performance1, query_performance2)
    # print(np.corrcoef(query_performance1, query_performance2))
    # # get_density_scatter(query_performances[-2], query_performances[-1])
    # # print(np.corrcoef(query_performances[-1], query_performances[-2]))
    # exit(0)      

    
    nsg_query_performance = resolve_performance_variance_log(nsg_query_performance_log_path)
    nsg_query_performances = [np.array(nsg_query_performance)]
    for i in range(9):
        nsg_query_performances.append(np.array(resolve_performance_variance_log(nsg_query_performance_log_paths[i])))
        print(nsg_query_performances[-1].shape, np.average(nsg_query_performances[-1]))
    nsg_query_performances = np.array(nsg_query_performances)
    nsg_query_performance_avg = np.average(nsg_query_performances, axis=0) 
    # half = int(nsg_query_performances.shape[0] / 2)
    # query_performance1 = np.average(nsg_query_performances[:half], axis=0)
    # query_performance2 = np.average(nsg_query_performances[half: ], axis=0)  
    # get_density_scatter(query_performance1, query_performance2)
    # print(np.corrcoef(query_performance1, query_performance2))
    # # get_density_scatter(nsg_query_performances[0], nsg_query_performances[1])
    # # print(np.corrcoef(nsg_query_performances[0], nsg_query_performances[1]))
    # exit(0)      

    
    kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance{recall:.2f}.log_clean')

    kgraph_query_performance = resolve_performance_variance_log(kgraph_query_performance_log_path)
    # in_ds_query_performance = resolve_performance_variance_log(in_ds_query_performance_log_path)
    # kgraph_query_performance2 = resolve_performance_variance_log(kgraph_query_performance_log_path2)
    
    # n_rknn = read_ibin_simple(ind_path)
    # ma_base_dist = read_fbin_simple(ma_base_dist_path)
    
    # kgraph_query_performance_hardest = resolve_performance_variance_log(kgraph_query_performance_log_path + '_hardest')
    # kgraph_query_performance_hardest2 = resolve_performance_variance_log(kgraph_query_performance_log_path + '2_hardest')
    
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
    b = query_performance_avg
    
    get_density_scatter(a,b)
    # calculate spearman correlation
    import scipy.stats as stats
    print(stats.spearmanr(a,b))
    print(np.corrcoef(a,b))
    
    # for i in range(len(query_performance_avg)):
    #     if kgraph_query_performance[i] < 5000 and query_performance_avg[i] > 5000: 
    #         print(f'{i}: {kgraph_query_performance[i]}, {query_performance_avg[i]}')