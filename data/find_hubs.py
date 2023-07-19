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
    
def read_ivecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
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
    
def ibin_read(fname: str):
    return np.fromfile(fname, dtype='int32')

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
dataset = 'deep'
ef = 500
M = 16

if __name__ == '__main__':
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    # read data vectors
    print(f"Reading {dataset} from {data_path}")
    X = read_fvecs(data_path)
    pos_path = os.path.join(path, f'{dataset}_pos2.ibin')
    print(f"Reading positions from {pos_path}")
    pos = ibin_read(pos_path)  # pos[new_id] = old_id
    # pos = np.arange(len(pos))
    data_shuf_path = os.path.join(path, f'{dataset}_base_shuf2.fvecs')
    print(f"Reading shuffled data from {data_shuf_path}")
    X_new = read_fvecs(data_shuf_path)
    ind_path = os.path.join(path, f'{dataset}_ind2.ibin')
    print(f"Reading indegree from {ind_path}")
    indegree = ibin_read(ind_path)
    
    origin2new = {}
    for i in range(len(pos)):
        origin2new[pos[i]] = i
    
    hubs = []
    for i in range(len(indegree)):
        if indegree[i] > 80:
            hubs.append((i, indegree[i]))
    hubs.sort(key=lambda x:x[1], reverse=True)
    hubs = np.array(hubs)[:, 0]
            
    X_new_hubs = X_new[hubs]
    
    trans_hubs = np.array(hubs)
    for i in range(len(hubs)):
        trans_hubs[i] = pos[hubs[i]]
        
    original_hubs_path = os.path.join(path, f'{dataset}_hubs.ivecs')
    print(f"Reading original hubs from {original_hubs_path}")
    original_hubs = read_ivecs(original_hubs_path).flatten()
    X_oh = X[original_hubs]
    trans_original_hubs = []
    for i in range(len(original_hubs)):
        trans_original_hubs.append(origin2new[original_hubs[i]])
    trans_original_hubs = np.array(trans_original_hubs)
    trans_original_hubs_ind = indegree[trans_original_hubs]
    
    to_ivecs(os.path.join(path, f'{dataset}_hubs2.ivecs'), hubs.reshape(-1, 1))
    print(f"nhub_points = {len(hubs)}")
    overlap = np.intersect1d(trans_hubs, original_hubs)
    print(f"#overlap = {len(overlap)}")
    
    