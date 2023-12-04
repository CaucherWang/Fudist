from utils import *

import numpy as np
import networkx as nx
import os

visited_points = None

def convert_graph2edge_list(graph, KG):
    edges = []
    num_points = len(graph)
    for point, neighbors in enumerate(graph):
        for neighbor in neighbors[:KG+1]:
            if neighbor != point:
                edges.append((point, neighbor))
    return edges

source = './data/'
dataset = 'deep'
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    graph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs')
    KG = 32
    ind_path = os.path.join(source, dataset, f'{dataset}_ind_{KG}.ibin')

    # X = read_fvecs(base_path)
    G = read_ivecs(graph_path)
    graph = convert_graph2edge_list(G, KG)
    print('convert finished')
    n_rknn = read_ibin_simple(ind_path)
    
    PRGraph = nx.DiGraph(graph)
    print('initialized finished')
    pagerank_values = nx.pagerank(PRGraph)
    print('pagerank finished')
    prvalues = np.array(list(pagerank_values.values()))
    
    print(np.corrcoef(prvalues, n_rknn))
    
    # plot the scatter point
    import matplotlib.pyplot as plt
    plt.plot(n_rknn, prvalues, 'o', color='black', markersize=0.1)
    plt.xlabel('indegree of kNN graph')
    plt.ylabel('prvalues')
    # plt.xlim(0, 1200)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title(f'{dataset} indegree correlation')
    plt.savefig(f'./figures/{dataset}-pagerank-hub-correlation.png')

    
    # Q = read_fvecs(query_path)[:1000]
    # GT = read_ivecs(GT_path)
    # n_rknn = get_reversed_knn_number(G, KG)
    # write_ibin_simple(ind_path, n_rknn)
    # 
    # # print(get_skewness(n_rknn))
    
    # hubs_percent = 10 * 0.01
    # hubs_num = int(hubs_percent * n_rknn.shape[0])
    # hubs = np.argsort(n_rknn)[-hubs_num:]
    # sum_inside = 0
    # sum_outside = 0
    # for hub in hubs:
    #     neighbors = G[hub][:KG+1].copy()
    #     neighbors = np.delete(neighbors, np.where(neighbors == hub))
    #     intersect = np.intersect1d(neighbors, hubs)
    #     sum_inside += len(intersect)
    #     sum_outside += len(neighbors) - len(intersect)
    # print(sum_inside, sum_outside, sum_inside + sum_outside, sum_inside / (sum_inside + sum_outside))
