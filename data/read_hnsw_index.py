import hnswlib
import numpy as np
import struct
import os
from utils import *
    

source = './data/'
dataset = 'glove1.2m'
ef = 500
M = 16

if __name__ == '__main__':
    path = os.path.join(source, dataset)
    index_path = os.path.join(path, f'{dataset}_ef{ef}_M{M}.index')
    print(f"Reading index from {index_path}.")
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    # read data vectors
    print(f"Reading {dataset} from {data_path}.")
    X = read_fvecs(data_path)

    
    index = read_hnsw_index(index_path, X.shape[1])
    data_size = index.get_current_count()
    print(f"totally {data_size} items in index")
    capacity = index.get_max_elements()
    print(f"capacity = {capacity}")
    id_list = index.get_ids_list()
    min_id = min(id_list)
    max_id = max(id_list)
    assert len(id_list) == data_size
    
    ann_data = index.getAnnData()
    data_level_0 = ann_data['data_level0']
    size_data_per_element = ann_data['size_data_per_element']
    offset_data = ann_data['offset_data']
    internal_ids = ann_data['label_lookup_internal']
    external_ids = ann_data['label_lookup_external']
    id2label = {}
    label2id = {}
    for i in range(len(internal_ids)):
        id2label[internal_ids[i]] = external_ids[i]
        label2id[external_ids[i]] = internal_ids[i]
        
    print('read index finished')
    
    
    
    # total_cnt = 0
    # nhub_points = 0
    # indegree = np.zeros(data_size, dtype=np.int32)
    # for i in range(data_size):
    #     if i % 100000 == 0:
    #         print(i)
    #     neighbors = get_neighbors_with_external_label(data_level_0, i, size_data_per_element, label2id)
    #     total_cnt += len(neighbors)
    #     for j in neighbors:
    #         indegree[j] += 1
    # hubs = []
    # for i in range(len(indegree)):
    #     if indegree[i] > 100:
    #         hubs.append(i)
    # hubs = np.array(hubs)
    # write_ivecs(os.path.join(path, f'{dataset}_hubs.ivecs'), hubs.reshape(-1, 1))
    # print(f"average out degree = {total_cnt / data_size}")
    # print(f"nhub_points = {len(hubs)}")
    
    # mean = np.mean(indegree)
    # percents = np.percentile(indegree, q=[5,10,25,50,75,90])
    # print(mean)
    # print(percents)
    

    
