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
dataset = 'glove1.2m'
ef = 500
M = 16

if __name__ == '__main__':
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    # read data vectors
    ind_path = os.path.join(path, f'{dataset}_ind.ibin')
    print(f"Reading indegree from {ind_path}")
    indegree = np.array(ibin_read(ind_path))
    
    means = np.mean(indegree)
    print(means)
    percentiles = np.percentile(indegree, [5,10, 25, 50, 75, 90, 95 , 99])
    print(percentiles)

    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.hist(indegree, bins=100, edgecolor = 'black')
    plt.title(f'Distrib. of indegrees of {dataset}, M={M}')
    plt.ylabel('Frequency')
    plt.xlabel('Indgree')
    plt.tight_layout()
    plt.savefig(f'./figures/hist_ind_{dataset}.png')
    
    