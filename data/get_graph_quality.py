import queue
from turtle import circle
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


fig = plt.figure(figsize=(6,4.8))

params = {
    'gauss100':{'M': 50, 'ef': 500, 'recall': 0.86,'L': 200, 'R': 100, 'C': 500},
    'rand100': {'M': 64, 'ef': 500, 'recall': 0.86,'L': 200, 'R': 100, 'C': 500},
    'deep': {'M': 16, 'ef': 500, 'recall': 0.98,  'L': 50, 'R': 32, 'C': 500},
    'sift': {'M': 16, 'ef': 500, 'recall': 0.98,  'L': 50, 'R': 32, 'C': 500},
}
source = './data/'
result_source = './results/'
dataset = 'rand100'
idx_postfix = '_plain'
Kbuild = 499
M = params[dataset]['M']
efConstruction = params[dataset]['ef']
R = params[dataset]['R']
L = params[dataset]['L']
C = params[dataset]['C']
target_recall = params[dataset]['recall']
target_prob = 0.96
select = 'kgraph'
if __name__ == "__main__":
    # graph_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_K{Kbuild}.nsw.index')
    index_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}.index{idx_postfix}')
    KG = 499
    
    kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')
    nsg_path = os.path.join(source, dataset, f'{dataset}_L{L}_R{R}_C{C}.nsg')
    standard_hnsw_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}_hnsw.ibin{idx_postfix}')

    # K = 500
    G = read_ivecs(kgraph_path)
    assert G.shape[1] >= 499
    
    hnsw = read_ibin(standard_hnsw_path)
    ep, nsg = read_nsg(nsg_path)
    
    for kgt in [50,100,200, 300, 400,499]:
        print(kgt, 'nsg' , get_graph_quality_list(nsg, G, kgt))
        print(kgt, 'hnsw', get_graph_quality(hnsw, G, kgt))
    
    
    
