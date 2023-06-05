import hnswlib
import numpy as np
import struct
import os
from sklearn.decomposition import PCA
from numba import njit

def ed(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def generate_matrix(lowdim, D):
    return np.random.normal(size=(lowdim, D))

def get_compact_sign_matrix(sgn_m, dt):
    return np.packbits(sgn_m, axis=1, bitorder='little').view(dt)

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

@njit
def get_dist_ground_truth(q_vec, c_vec, d_vec, P):
    q_proj_vec = (q_vec @ c_vec) / (c_vec @ c_vec) * c_vec
    d_proj_vec = (d_vec @ c_vec) / (c_vec @ c_vec) * c_vec
    proj_delta = q_proj_vec - d_proj_vec
    part1 = proj_delta @ proj_delta

    d_res_vec = d_vec - d_proj_vec
    part2 = d_res_vec @ d_res_vec

    q_res_vec = q_vec - q_proj_vec
    part3 = q_res_vec @ q_res_vec

    # part4 = -2 * d_res_vec @ q_res_vec
    part4_ = -2 * np.sqrt(part2) * np.sqrt(part3) * np.cos(np.sum(np.sign(q_res_vec @ P) != np.sign(d_res_vec @ P)) / P.shape[1] * np.pi)

    return [part1, part2, part3, part4_], part1 + part2 + part3 + part4_

@njit
def finger_dist(t, b, d_res, c_2, q_res, q_res_2, sgn_q_res_P, sgn_d_res_P):
    part1 = (t-b) * (t-b) * c_2

    part2 = d_res * d_res

    part3 = q_res_2

    cos = np.cos(np.sum(sgn_q_res_P != sgn_d_res_P) / len(sgn_d_res_P) * np.pi)
    part4 = -2 * q_res * d_res * cos

    # return [part1, part2, part3, part4], part1 + part2 + part3 + part4
    return part1 + part2 + part3 + part4

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

def to_floats(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            a = struct.pack('f', y)
            fp.write(a)


# return internal ids
def get_neighbors_with_external_label(data_level_0, external_label, size_data_per_element, label2id):
    internal_id = label2id[external_label]
    return get_neighbors_with_internal_id(data_level_0, internal_id, size_data_per_element)


source = './data'
datasets_map = {
    # 'imagenet': (16,6, 16),   
    # 'msong': (8,6),
    # 'word2vec': (48, 6, 16),
    # 'ukbench': (16,8),
    # 'deep': (16, 8, 16),
    'gist': (16, 8, 96),
    # 'glove1.2m': (128, 8, 20),
    # 'sift': (16, 8, 16),
    # 'tiny5m': (48, 8, 16),
    # 'uqv':(16, 8, 16),
    # 'glove-100':(16, 4, 16),
    # 'crawl': (16, 6, 16),
    # 'enron':(16, 8, 64),
    # 'mnist':(8, 8, 64),
    # 'cifar': (8,8,64),
    # 'sun': (8, 8, 64),
    # 'trevi':(16, 8, 64),
    # 'notre':(8, 8, 16),
    # 'nuswide':(48, 10, 64),
}

ef = 500
M = 16

lsh_dim = 64

if __name__ == '__main__':
    for dataset in datasets_map.keys():
        M = datasets_map[dataset][0]
        path = os.path.join(source, dataset)

        projection_path = os.path.join(path, f'FINGER_{dataset}M{M}ef{ef}_LSH_{lsh_dim}.fvecs')
        b_dres_path = os.path.join(path, f'FINGER_{dataset}M{M}ef{ef}_b_dres.fvecs')
        sgn_dres_P_path = os.path.join(path, f'FINGER_{dataset}M{M}ef{ef}_sgn_dres_P.ivecs')
        c_2_path = os.path.join(path, f'FINGER_{dataset}M{M}ef{ef}_c_2.fvecs')
        c_P_path = os.path.join(path, f'FINGER_{dataset}M{M}ef{ef}_c_P.fvecs')
        start_idx_path = os.path.join(path, f'FINGER_{dataset}M{M}ef{ef}_start_idx.ivecs')
        index_path = os.path.join(path, f'{dataset}_ef{ef}_M{M}.index')

        print(f"Loading data from {dataset}")
        
        P = read_fvecs(projection_path)
        edge_info_float = read_fvecs(b_dres_path)
        edge_info_uint = read_ivecs(sgn_dres_P_path)
        c_2s = read_fvecs(c_2_path).reshape(-1)
        node_info_float = read_fvecs(c_P_path)
        startIdx = read_ivecs(start_idx_path).reshape(-1)

        
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        Q = read_fvecs(query_path)
        
        index = read_hnsw_index(index_path, X.shape[1])
        # index = read_hnsw_index(index_path, 961)

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
        
        print("Dataloaded.")

        query_sample_num = 1000
        data_sample_num = 1000

        q_num = min(Q.shape[0], query_sample_num)
        d_num = min(X.shape[0], data_sample_num)

        # get_dist_ground_truth(q_vec, c_vec, d_vec, P)
        
        result = []
        for q_label in range(q_num):
            if (q_label + 1) % 100 == 0:
                print(f'{q_label+1} / {q_num}')

            q_vec = Q[q_label]
            q_2 = q_vec @ q_vec

            for c_label in range(d_num):
                c_vec = X[c_label]
                c_2 = c_2s[c_label]
                assert(abs(c_vec@c_vec - c_2) < 1e-5 )
                c_P = node_info_float[c_label]
                c_start = startIdx[c_label]

                old_dist = ed(Q[q_label], X[c_label])
                old_dist = old_dist * old_dist

                t = (q_2 + c_2 - old_dist) / 2 / c_2
                q_res_2 = q_2 - t * t * c_2
                q_res = np.sqrt([q_res_2,0][int(q_res_2<0)])
                q_res_P = q_vec @ P - t * c_P
                sgn_q_res_P = (np.sign(q_res_P) > 0)

                neighbors = get_neighbors_with_external_label(data_level_0, c_label, size_data_per_element, label2id)
                for d_offset, nei in enumerate(neighbors):
                    nei_label = id2label[nei]
                    d_vec = X[nei_label]

                    b = edge_info_float[c_start + d_offset][0]
                    d_res = edge_info_float[c_start + d_offset][1]
                    sgn_d_res_P = edge_info_uint[c_start + d_offset]
                    
                    real_dist = np.sum((Q[q_label] - X[nei_label]) ** 2)
                    # info, dist = get_dist_ground_truth(Q[q_label], X[c_label], X[nei_label], P)
                    dist = finger_dist(t, b, d_res, c_2, q_res, q_res_2, sgn_q_res_P, sgn_d_res_P)
                    # print(np.array(info) - np.array(info2))
                    # print(dist - dist2)
                    if real_dist == 0:
                        if dist == 0:
                            result.append(1)
                    else:
                        result.append(dist / real_dist)
        print(np.mean(result))
        result_path = os.path.join(path, f'FINGER_approx_dist.floats')
        to_floats(result_path, result)


    # for dataset in datasets_map.keys():
    #     M = datasets_map[dataset][0]
    #     path = os.path.join(source, dataset)
    #     index_path = os.path.join(path, f'{dataset}_ef{ef}_M{M}.index')

    #     data_path = os.path.join(path, f'{dataset}_base.fvecs')
    #     # read data vectors
    #     print(f"Reading {dataset} from {data_path}.")
    #     X = read_fvecs(data_path)

    #     index = read_hnsw_index(index_path, 961)

    #     ann_data = index.getAnnData()
    #     data_level_0 = ann_data['data_level0']
    #     size_data_per_element = ann_data['size_data_per_element']
    #     offset_data = ann_data['offset_data']
    #     internal_ids = ann_data['label_lookup_internal']
    #     external_ids = ann_data['label_lookup_external']
    #     id2label = {}
    #     label2id = {}
    #     for i in range(len(internal_ids)):
    #         id2label[internal_ids[i]] = external_ids[i]
    #         label2id[external_ids[i]] = internal_ids[i]

    #     # loop1: get nei info of each node
    #     print("Getting neighbor's info of each node")

    #     data_size = X.shape[0]

    #     # data_size = 100
    #     total_cnt = 0
    #     startIdx = np.zeros(data_size, dtype=np.int32)

    #     for i in range(data_size):
    #         # print(i)
    #         startIdx[i] = total_cnt
    #         neighbors = get_neighbors_with_external_label(data_level_0, i, size_data_per_element, label2id)
    #         total_cnt += len(neighbors)

    #     # loop2: construct the d res
    #     print("Constrcuting vector of d_res")

    #     d_res_vecs = []
    #     for i in np.random.choice(data_size, 100000):
    #     # for i in range(data_size):
    #         neighbors = get_neighbors_with_external_label(data_level_0, i, size_data_per_element, label2id)
    #         nei_labels = [id2label[nei] for nei in neighbors]
    #         nei_vecs = X[nei_labels]
    #         c_2 = X[i] @ X[i]
    #         if c_2 != 0:
    #             nei_proj_vecs = np.outer((nei_vecs @ X[i]) / c_2, X[i])
    #             nei_res_vecs = nei_vecs - nei_proj_vecs
    #             d_res_vecs += list(nei_res_vecs)
    #     # generate P with eigenvector of d res vectors

    #     print('Getting PCA of d_res')
    #     pca = PCA(n_components=lsh_dim)
    #     pca.fit(d_res_vecs)
    #     P = pca.components_.T


    #     print('Begin main preprocess')
    #     # |c|^2
    #     # c_2 = np.sum(X * X, axis = 1)
    #     c_2s = np.zeros(data_size, dtype=np.float32)
    #     node_info_float = np.zeros((data_size, lsh_dim), dtype=np.float32)

    #     # b, d_res
    #     edge_info_float = np.zeros((total_cnt, 2), dtype=np.float32)

    #     # sgn array of lsh(d_res)
    #     # edge_info_uint = np.zeros((total_cnt, num_int_lsh), dtype=np.uint32)
    #     edge_info_uint = np.zeros((total_cnt, lsh_dim), dtype=np.uint32)


    #     cur_idx = 0
    #     for cur_label in range(data_size):
    #         assert(cur_idx == startIdx[cur_label])
    #         cur_vec = X[cur_label]
    #         cur_c_2 = cur_vec @ cur_vec
    #         c_2s[cur_label] = cur_c_2
    #         neighbors = get_neighbors_with_external_label(data_level_0, cur_label, size_data_per_element, label2id)
    #         num_nei = len(neighbors)

    #         c_P = cur_vec @ P
    #         node_info_float[cur_label, :] = c_P
    #         # print(cur_label)

    #         if cur_c_2 == 0:
    #             edge_info_float[cur_idx: cur_idx+num_nei, :] = np.zeros((num_nei, 2))
    #         else:
    #             nei_labels = [id2label[nei] for nei in neighbors]
    #             nei_vecs = X[nei_labels]
    #             # print(nei_vecs.shape)

    #             bs = nei_vecs @ cur_vec / cur_c_2
                
    #             # cur_vec 

    #             d_2s = np.sum(nei_vecs * nei_vecs, axis=1, dtype=np.float32)
    #             d_proj_2s = bs * bs * cur_c_2
    #             d_res_2s = d_2s - d_proj_2s
    #             if np.any(d_res_2s < 0):
    #                 d_res_2s[d_res_2s < 0] = 0
    #             d_ress = np.sqrt(d_res_2s)

    #             edge_info_float[cur_idx: cur_idx+num_nei, :] = np.array([bs, d_ress]).T

                
    #             # use little bit order to compact sign array. (might need changing the order according to OS)
    #             # two examples assuming 4 bits per compact num: 
    #             #   1, 0, 1, 0 -> 0b0101;
    #             #   1, 0, 1, 0 | 1, 1, 1, 0 -> 0b0101, 0b0111
                
    #             d_res_vecs_dot_P = nei_vecs @ P - np.outer(bs, cur_vec) @ P
    #             edge_info_uint[cur_idx: cur_idx+num_nei, :] = (np.sign(d_res_vecs_dot_P) > 0)
                
    #             # edge_info_uint[cur_idx: cur_idx+num_nei, :] = get_compact_sign_matrix(np.sign(d_res_vecs_dot_P)>0, binary_view_type)
                
    #         cur_idx += num_nei

    #     # Stored in file
    #     projection_path = os.path.join(path, f'FINGER_{dataset}M{M}ef{ef}_LSH_{lsh_dim}.fvecs')
    #     b_dres_path = os.path.join(path, f'FINGER_{dataset}M{M}ef{ef}_b_dres.fvecs')
    #     sgn_dres_P_path = os.path.join(path, f'FINGER_{dataset}M{M}ef{ef}_sgn_dres_P.ivecs')
    #     c_2_path = os.path.join(path, f'FINGER_{dataset}M{M}ef{ef}_c_2.fvecs')
    #     c_P_path = os.path.join(path, f'FINGER_{dataset}M{M}ef{ef}_c_P.fvecs')
    #     start_idx_path = os.path.join(path, f'FINGER_{dataset}M{M}ef{ef}_start_idx.ivecs')

    #     to_fvecs(projection_path, P)
    #     to_fvecs(b_dres_path, edge_info_float)
    #     to_ivecs(sgn_dres_P_path, edge_info_uint)
    #     to_fvecs(c_2_path, c_2s.astype(np.float32).reshape(data_size,1))
    #     to_fvecs(c_P_path, node_info_float)
    #     to_ivecs(start_idx_path, startIdx.reshape(data_size,1))