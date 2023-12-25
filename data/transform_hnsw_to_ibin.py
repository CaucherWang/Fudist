from utils import *
from queue import PriorityQueue
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density # adds projection='scatter_density'


fig = plt.figure(figsize=(6,4.8))
          
source = './data/'
result_source = './results/'
dataset = 'rand100'
idx_postfix = '_plain'
shuf_postfix = ''
efConstruction = 2000
Kbuild = 16
M = 140
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    # graph_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_K{Kbuild}.nsw.index')
    index_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}.index{idx_postfix}')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    standard_hnsw_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}_hnsw.ibin{idx_postfix}')
    
    
    Q = read_fvecs(query_path)
    hnsw = read_hnsw_index_aligned(index_path, Q.shape[1])
    new_hnsw = shuffled_hnsw_to_standard_form(hnsw, np.arange(len(hnsw)), M)
    write_ibin(standard_hnsw_path, new_hnsw)

    # for i in range(3, 10):
    #     pos_path = (os.path.join(source, dataset, f'{dataset}_shuf{i}.ibin'))
    #     hnsw_path = (os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}.index_plain_shuf{i}'))
    #     new2old = read_ibin_simple(pos_path)
    #     hnsw = read_hnsw_index_aligned(hnsw_path, Q.shape[1])
    #     new_hnsw = shuffled_hnsw_to_standard_form(hnsw, new2old, M)
    #     write_ibin(standard_hnsw_path + f'_shuf{i}', new_hnsw)



