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



source = './data/'
result_source = './results/'
dataset = 'deep'
kgraph_od = '_2048'
if __name__ == "__main__":

    kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth{kgraph_od}.ivecs')

    KGraph = read_ivecs(kgraph_path)
    KGraph_clean = clean_kgraph(KGraph)
    write_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth{kgraph_od}.ivecs_clean'), KGraph_clean)

    # KGraph = read_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth{kgraph_od}.ivecs_clean'))[:, :Kbuild]
    # another_kgraph = read_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs_clean'))
    # diff = np.where(KGraph != another_kgraph)
    # print(diff[0])
    # real_Diff_index = []
    # i = 0
    # while i < diff[0].shape[0]:
    #     if diff[0][i] != diff[0][i+1]:
    #         real_Diff_index.append((diff[0][i], diff[1][i]))
    #         i += 1
    #     else:
    #         i += 2
    # print(real_Diff_index)
    # revG = get_reversed_graph_list(KGraph)
    # write_obj(reversed_kgraph_path, revG)
    # # revG = read_obj(reversed_kgraph_path)
    # exit(0)
