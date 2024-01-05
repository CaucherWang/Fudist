from types import new_class
import numpy as np
import os
from utils import *

def rand_select(nq, nx):
    # randomly select nq numbers from [0,nx)
    return np.random.choice(nx, nq, replace=False)


def update_gt(gt, original_positions):
    # update the ground truth
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            gt[i][j] = original_positions[gt[i][j]]
    return gt

@njit(parallel=True)
def update_kgraph(kgraph, old2new):
    # update the kgraph
    new_graph = np.zeros(kgraph.shape, dtype=np.int32)
    for i in range(kgraph.shape[0]):
        if i % 100000 == 0:
            print(i)
        cur_row = old2new[i]
        for j in range(kgraph.shape[1]):
            new_graph[cur_row][j] = old2new[kgraph[i][j]]
    return new_graph

source = './data/'
datasets = ['gist']
# shuf_num = 4
if __name__ == '__main__':
    for dataset in datasets:
        dir = os.path.join(source, dataset)
        kgraph_path = os.path.join(dir, f'{dataset}.efanna')
        G = read_ivecs(kgraph_path)
        rows = G.shape[0]

        for shuf_num in range(3, 30):
            pos_file = os.path.join(dir, f'{dataset}_shuf{shuf_num}.ibin')
            # kgraph_path = os.path.join(dir, f'{dataset}_self_groundtruth.ivecs_clean')
            # new_kgraph_path = os.path.join(dir, f'{dataset}_self_groundtruth.ivecs_clean_shuf{shuf_num}')
            
            new_kgraph_path = os.path.join(dir, f'{dataset}.efanna_shuf{shuf_num}')

            # Create an array of indices representing the original positions
            original_positions = read_ibin_simple(pos_file)
            old2new = [ 0 for i in range(rows) ]
            for i in range(rows):
                old2new[original_positions[i]] = i
                
            new_graph = update_kgraph(G, old2new)
            write_ivecs(new_kgraph_path, new_graph)
        