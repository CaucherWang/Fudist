import queue
from turtle import circle
from numpy import percentile, sort
from requests import get
from sklearn.utils import deprecated
from sympy import Q
from utils import *
from queue import PriorityQueue
import os
from unionfind import UnionFind
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density # adds projection='scatter_density'

# @njit(parallel=True)
def get_mrng_positions(mrng, kgraph):
    mrng_edges_positions = []
    for i in range(len(mrng)):
        mrng_edges_positions.append([])
    for i in range(len(mrng)):
        # if i % 10000 == 0:
        #     print(f'processing {i}th node')
        tmp = []
        index_i = 0
        index_j = 0
        while index_i < len(mrng[i]):
            while index_j < len(kgraph[i]) and kgraph[i][index_j] != mrng[i][index_i]:
                index_j += 1
            tmp.append(index_j)
            index_i += 1
        mrng_edges_positions[i] = tmp
    
    # flatten the 2-d list to 1-d list
    # mrng_edges_positions = [item for sublist in mrng_edges_positions for item in sublist]
    return mrng_edges_positions

params = {
    'gauss100':{'M': 100, 'ef': 2000, 'L': 200, 'R': 100, 'C': 500, 'recall':0.86},
    'rand100': {'M': 140, 'ef': 2000, 'L': 200, 'R': 100, 'C': 500, 'recall':0.86},
    'deep': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500, 'recall':0.98},
    'sift': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500, 'recall':0.98},
}
source = './data/'
result_source = './results/'
dataset = 'rand100'
target_prob = 0.96
idx_postfix = '_clean'
efConstruction = params[dataset]['ef']
M = params[dataset]['M']
L = params[dataset]['L']
R = params[dataset]['R']
C = params[dataset]['C']
target_recall = params[dataset]['recall']
KMRNG = 9999
select = 'kgraph'
if __name__ == "__main__":

    # get the positions of edges in mrng from kgraph
    mrng_edges_positions = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_positions_distribution.ibin_mrng'))
            
    sorted_mrng_edge_positions = sort(np.array(mrng_edges_positions))
    # y-axis is the portion of the positions
    # x-axis is the upper bound of the positions
    percentiles = np.linspace(0, 100, 100)
    x = []
    for i in range(len(percentiles)):
        x.append(percentile(sorted_mrng_edge_positions, percentiles[i]))
    plt.plot(x, percentiles, **plot_info[0])
    plt.xlabel('upper bound of the positions')
    plt.ylabel('portion of the positions')
    plt.tight_layout()
    plt.savefig(f'./figures/{dataset}/{dataset}-mrng-kgraph-edge-position.png')
    print(f'save to figure ./figures/{dataset}/{dataset}-mrng-kgraph-edge-position.png')
    
    
            
    