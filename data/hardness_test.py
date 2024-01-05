import queue
from utils import *
from queue import PriorityQueue
import os
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
    # plt.xlabel('k-occurrence (500)')
    # plt.xlabel('Query difficulty')
    # plt.xlabel('LID')
    # plt.xlabel('reach_delta_0_rigorous')
    # plt.xlabel('metric for dynamic bound')
    # plt.xlabel('delta_0')
    # plt.xlabel(r'$K_0^{0.96}@0.98$')
    # plt.xlabel(r'$\delta_0^{\forall}-g@0.98$')
    # plt.xlabel(r'$\Sigma ME^{0.96}_{\delta_0}$@0.98')
    # plt.xlabel(r'$ME^{0.96}_{\delta_0}$@0.98-exhausted (MRNG)')
    plt.xlabel(r'$ME^{0.86}_{\delta_0}$@0.98-exhausted (MRNG)')
    # plt.xlabel(r'$Area(ME^{0.96}_{\delta_0}$@0.98)-Interpolation')
    # plt.xlabel(r'$Area(ME^{0.96}_{\delta_0}$@0.98)-Regression')
    # plt.xlabel(r'$ME^{0.96}_{\delta_0}$@0.98')
    # plt.xlabel(r'$Kgraph-100-\hat{ME^{0.96}_{\delta_0}}$@0.90-exhausted')
    # plt.xlabel(r'$ME^{\forall}_{\delta_0}-reach$@0.98')
    # plt.ylabel(r'NDC (recall@50>0.98)')
    # plt.ylabel(r'KGraph(100) $LQC_{50}^{0.98}$')
    # plt.xlabel('local intrinsic dimensionality')
    plt.ylabel(r'HNSW(32) $LQC_{50}^{0.98}$')
    # plt.ylabel('1NN distance')
    # plt.tight_layout()
    # plt.xlim(0, 1.5*1e7)
    # plt.ylim(0, 20000)
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

params = {
    'gauss100':{'M': 160, 'ef': 2000, 'L': 500, 'R': 200, 'C': 2000, 
                'recall':0.86, 'KMRNG':2047,
                'rp':{
                    0.86: { # 0.60
                        'prob': 0.86,
                        'lognum': 21,
                        'nsglognum':21
                    },
                    0.94: { # 0.50
                        'prob':0.94,
                        'lognum': 21
                    }
                }
                },
    'rand100': {'M': 140, 'ef': 2000, 'L': 200, 'R': 100, 'C': 500, 
                'recall':0.86, 'KMRNG':2047,
                'rp':{
                    0.86: { # 0.68
                        'prob': 0.86,
                        'lognum':21
                    },
                    0.94: { #0.59
                        'prob':0.90,
                        'lognum': 27
                    }
                }
                },
    'deep': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500, 
             'recall':0.98, 'KMRNG':2047, 
             'rp':{
                 0.98:{ # 0.85
                     'prob':0.98,
                     'lognum':7
                 },
                 0.92:{ # 0.92
                        'prob':0.96,
                        'lognum':7
                    },
                 }
             }, 
    'sift': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500, 
             'recall':0.92, 'KMRNG':2047,
             'rp':{
                0.98:{ # 0.87
                    'prob':0.98,
                    'lognum':7
                }, 
                0.92:{ # 0.90
                    'prob':0.94,
                    'lognum':7
                }
             }
             },
}
source = './data/'
result_source = './results/'
dataset = 'gauss100'
target_recall = params[dataset]['recall']
target_prob = params[dataset]['rp'][target_recall]['prob']
idx_postfix = '_plain'
efConstruction = params[dataset]['ef']
M=params[dataset]['M']
L = params[dataset]['L']
R = params[dataset]['R']
C = params[dataset]['C']
KMRNG = params[dataset]['KMRNG']

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
    # delta_result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_delta.log')
    # delta0_point_path = os.path.join(source, dataset, f'{dataset}_delta0_point_K{Kbuild}.ibin')
    # self_delta0_max_knn_rscc_point_recall_path = os.path.join(source, dataset, f'{dataset}_self_delta0_max_knn_rscc_point_recall{target_recall}_K{Kbuild}.ibin')
    # delta0_rigorous_point_path = os.path.join(source, dataset, f'{dataset}_delta0_rigorous_point_K{Kbuild}.ibin')
    # delta0_max_knn_rscc_point_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_poin_K{Kbuild}.ibin')
    # kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log')
    # in_ds_kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log_in-dataset')
    # in_ds_kgraph_query_performance_recall_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance{target_recall}.log_in-dataset')
    query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance{target_recall}.log_plain')
    query_performance_log_paths = []
    nsg_query_performance_log_path = os.path.join(result_source, dataset, f'{dataset}_nsg_L{L}_R{R}_C{C}_perform_variance{target_recall:.2f}.log')
    query_performance_log_paths = []
    nsg_query_performance_log_paths = []
    for i in range(3, 30):
        nsg_query_performance_log_paths.append(os.path.join(result_source, dataset, f'{dataset}_nsg_L{L}_R{R}_C{C}_perform_variance{target_recall:.2f}.log_shuf{i}'))
    for i in range(3, 30):
        query_performance_log_paths.append(os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance{target_recall}.log_plain_shuf{i}'))

    # tmp = resolve_performance_variance_log_multi(os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance{target_recall}.log_plain_shuf23'))  
    # get_density_scatter(tmp[0], tmp[1])
    # exit(0)
    
    # delta0_point_path = os.path.join(source, dataset, f'{dataset}_delta0_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                  '_K2047.ibin_mrng')
    # delta0_point = read_ibin_simple(delta0_point_path)
    # delta0_point = np.array(delta0_point)
    # print(delta0_point.shape, np.max(delta0_point), np.min(delta0_point))
    # exit(0)

    # query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                                f'_delta_point391_K{KMRNG}.ibin_mrng'))

    # query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                                f'_K{KMRNG}_alpha60.ibin_ssg'))


    query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
                                                   f'_K{KMRNG}.ibin_mrng'))

    # query_hardness = resolve_performance_variance_log(os.path.join(result_source, dataset, f'SIMD_{dataset}_MRNG_K{KMRNG}_perform_variance{target_recall:.2f}.log'))

    # query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_K0_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                                f'_beta0.20.ibin_clean'))

    
    # query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                                '_K100.ibin_clean'))
    
    # query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                                '_K100.ibin_clean'))
    
    # GT_dist = read_fvecs(GT_dist_path)
    # query_hardness = get_lids(GT_dist[:, :50], 50)
    
    # KLIST = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375 ,400, 425, 450, 475, 499]
    # # KLIST = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375 ,400]
    # # # KLIST = [175, 200, 250, 300, 400, 499]
    # me_exhausted_path = os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                  '_K%d.ibin_clean')
    # # # me_exhausted_path = os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    # # #                                  '_delta_point500_K%d.ibin_clean')

    # me_exhausted = []
    # for K in KLIST:
    #     me_exhausted.append(read_ibin_simple(me_exhausted_path % K))
    # me_exhausted = np.array(me_exhausted).T
    # # query_hardness = np.average(me_exhausted, axis=0)
    # from scipy import interpolate
    # areas = []
    # for sample in me_exhausted:
    #     f = interpolate.interp1d(KLIST, sample)
    #     xnew = np.linspace(min(KLIST), max(KLIST), num=1000, endpoint=True)
    #     ynew = f(xnew)
    #     area = np.trapz(ynew, xnew)
    #     areas.append(area)
    
    # query_hardness = np.array(areas)

    # areas = []
    # for sample in me_exhausted:
    #     # Fit a polynomial of degree 2 to the data
    #     coefficients = np.polyfit(KLIST, sample, 2)
    #     polynomial = np.poly1d(coefficients)

    #     # Integrate the polynomial and evaluate it at the endpoints of the interval
    #     integral = polynomial.integ()
    #     area = integral(max(KLIST)) - integral(min(KLIST))
    #     areas.append(area)

    # query_hardness = np.array(areas)


    
    # K = 500
    # G = read_ivecs(kgraph_path)
    # ind_path = os.path.join(source, dataset, f'{dataset}_ind_{K}.ibin')
    # # n_rknn = get_reversed_knn_number(G, K)
    # # write_ibin_simple(ind_path, n_rknn)
    # n_rknn = read_ibin_simple(ind_path)
    # GT = read_ivecs(GT_path)
    # k_occur = get_query_k_occur(GT, n_rknn)
    
    # GT_dist = read_fvecs(GT_dist_path)
    # q_lids = get_lids(GT_dist[:, :50], 50)
    
    # hnsw = read_ibin(standard_hnsw_path)
    # print(get_graph_quality(hnsw, G, 500))
    # exit(0)
    
    # remove 750 and 9092 elements in query hardness
    query_hardness = np.delete(query_hardness, [2207, 5153])
    
    
    
    # query_performance = np.array(resolve_performance_variance_log(query_performance_log_path))
    # query_performances = [query_performance]
    # print(query_performances[-1].shape, np.average(query_performances[-1]))
    # for i in range(params[dataset]['rp'][target_recall]['lognum']):
    #     query_performances.append(np.array(resolve_performance_variance_log(query_performance_log_paths[i])))
    #     print(query_performances[-1].shape, np.average(query_performances[-1]))
    # query_performances = np.array(query_performances)
    # query_performance_avg = np.average(query_performances, axis=0)
    # # half = int(query_performances.shape[0] / 2)
    # # query_performance1 = np.average(query_performances[:half], axis=0)
    # # query_performance2 = np.average(query_performances[half: ], axis=0)        
    # # kgraph_query_performance_recall_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance{target_recall}.log_clean')
    # # kgraph_query_performance_recall = np.array(resolve_performance_variance_log(kgraph_query_performance_recall_log_path))
    

    nsg_query_performance = resolve_performance_variance_log(nsg_query_performance_log_path)
    nsg_query_performances = [np.array(nsg_query_performance)]
    for i in range(params[dataset]['rp'][target_recall]['lognum']):
        nsg_query_performances.append(np.array(resolve_performance_variance_log(nsg_query_performance_log_paths[i])))
        print(nsg_query_performances[-1].shape, np.average(nsg_query_performances[-1]))
    nsg_query_performances = np.array(nsg_query_performances)
    nsg_query_performance_avg = np.average(nsg_query_performances, axis=0) 
    query_performance_avg = nsg_query_performance_avg
    
    # kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K100_perform_variance{target_recall:.2f}.log_clean')

    # kgraph_query_performance = resolve_performance_variance_log(kgraph_query_performance_log_path)

    query_performance_avg = np.delete(query_performance_avg, [2207, 5153])

    # print(np.corrcoef(query_hardness, nsg_query_performance_avg))
    print(np.corrcoef(query_hardness, query_performance_avg))
    # print(np.corrcoef(query_hardness, kgraph_query_performance))

    # get_density_scatter(query_performances[0], query_performances[1])
    # get_density_scatter(query_performance1, query_performance2)
    get_density_scatter(query_hardness, query_performance_avg)
    # get_density_scatter(query_hardness, nsg_query_performance_avg)
    # get_density_scatter(query_hardness, kgraph_query_performance)
    # get_density_scatter(query_hardness, me)
    
    

    # get_density_scatter(query_hardness, query_performance)
