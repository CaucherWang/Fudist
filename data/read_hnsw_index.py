import hnswlib
import numpy as np
import struct
import os
def read_hnsw_index(filepath, D):
    '''
    Read hnsw index from binary file
    '''
    index = hnswlib.Index(space='l2', dim=D)
    index.load_index(filepath)
    return index

def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv
    
def to_ivecs(filename: str, array: np.ndarray):
    print(f"Writing File - {filename}")
    topk = (np.int32)(array.shape[1])
    array = array.astype(np.int32)
    topk_array = np.array([topk] * len(array), dtype='int32')
    file_out = np.column_stack((topk_array, array.view('int32')))
    file_out.tofile(filename)
    

def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

# return internal ids           
def get_neighbors_with_internal_id(data_level_0, internal_id, size_data_per_element):
    start = int(internal_id * size_data_per_element / 4)
    cnt = data_level_0[start]
    neighbors = []
    for i in range(cnt):
        neighbors.append(data_level_0[start + i + 1])
    return neighbors

# return internal ids
def get_neighbors_with_external_label(data_level_0, external_label, size_data_per_element, label2id):
    internal_id = label2id[external_label]
    return get_neighbors_with_internal_id(data_level_0, internal_id, size_data_per_element)

source = './data/'
dataset = 'gist'
ef = 500
M = 16

if __name__ == '__main__':
    path = os.path.join(source, dataset)
    index_path = os.path.join(path, f'DWT_{dataset}_ef{ef}_M{M}.index')
    
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    # read data vectors
    print(f"Reading {dataset} from {data_path}.")
    X = read_fvecs(data_path)

    
    index = read_hnsw_index(index_path, 961)
    cnt = index.get_current_count()
    print(f"totally {cnt} items in index")
    capacity = index.get_max_elements()
    print(f"capaciuty = {capacity}")
    id_list = index.get_ids_list()
    min_id = min(id_list)
    max_id = max(id_list)
    assert len(id_list) == cnt
    
    ids = np.array([0,1,2])
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
    
    
    for i in range(10):
        vec = X[i]  # or vec = index.get_items(np.array([i]))[0]
        neighbors = get_neighbors_with_external_label(data_level_0, i, size_data_per_element, label2id)
        print(len(neighbors))
        for neighbor in neighbors:
            label = id2label[neighbor]
            neighbor_vec = X[label]
    

    print(index.get_max_elements())
    
