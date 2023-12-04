from utils import *
from queue import PriorityQueue
import os
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,4.8))

def read_nsw(filename, numVectors):
    data = np.memmap(filename, dtype='uint64', mode='r')
    print(f'edge number * 2 = {len(data) - numVectors}')
    vectors = []
    cur_pos = 0
    for i in range(numVectors):
        vectorDim = int(data[cur_pos])
        vector = data[cur_pos + 1: cur_pos + 1 + vectorDim]
        vectors.append(vector)
        cur_pos += 1 + vectorDim
    return vectors
    # with open(filename, 'rb') as f:
    #     vectors = []
    #     for _ in range(numVectors):
    #         vectorDim = np.frombuffer(f.read(8), dtype=np.uint64)[0]    # size_t = unsigned long
    #         vector = np.frombuffer(f.read(vectorDim * 8), dtype=np.uint64)
    #         vectors.append(vector)

    # return vectors

class node:
    def __init__(self, id, dist):
        self.id = id
        self.dist = dist
    
    def __lt__(self, other):
        # further points are prior
        return self.dist > other.dist
    
    def __str__(self) -> str:
        return f'id: {self.id}, dist: {self.dist}'
    
def select_ep(n):
    return np.random.choice(n, 1)

def search(graph, vec, query, k, efsearch, visited_points=None, glanced_points=None, eps=None, ndc=0, nhops=0):
    resultSet = PriorityQueue(efsearch)     # len < efSearch
    candidateSet = PriorityQueue()  # unlimited lenth
    visited = set()
    if eps is None:
        eps = select_ep(vec.shape[0])
    for ep in eps:
        candidateSet.put(node(ep, -euclidean_distance(vec[ep], query)))
        
    while not candidateSet.empty():
        top = candidateSet.get()
        if not resultSet.empty() and -top.dist > resultSet.queue[0].dist:
            break
        nhops +=1
        if visited_points is not None:
            visited_points[top.id] += 1
        for nei in graph[top.id]:
            if nei in visited:
                continue
            visited.add(nei)
            if glanced_points is not None:
                glanced_points[nei] += 1
            ndc += 1
            dist = euclidean_distance(vec[nei], query)
            if not resultSet.full() or dist < resultSet.queue[0].dist:
                candidateSet.put(node(nei, -dist))
                if resultSet.full():
                    resultSet.get()
                resultSet.put(node(nei, dist))
                
    while resultSet._qsize() > k:
        resultSet.get()
        
    ret = []
    while not resultSet.empty():
        top = resultSet.get()
        ret.append([top.id, top.dist])
    ret.reverse()
    return ret, ndc, nhops
          
def do_expr(X, G, Q, GT, k, n_rknn, lengths, indegree, file = None):
    f = None 
    if file is not None:
        f = open(file, 'a')
        print(f'write to {file}')
    # efss = [500]
    efss = [60,80, 100, 150, 200, 300, 400, 500, 600, 750, 1000, 1500, 2000]
    # special_q = 1983
    # efss = [100, 200, 300, 400, 500]
    # efss = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 4000, 5000, 5200, 5400, 5600, 5800, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000, 12000, 15000, 18000, 20000, 25000, 30000, 35000, 40000]
    # efss = [50000, 60000, 80000, 100000]
    for efs in efss:
        visited_points = np.zeros(X.shape[0], dtype=np.int32)
        glanced_points = np.zeros(X.shape[0], dtype=np.int32)
        if f is None:
            print(f'efsearch: {efs}', end='\t')
        else:
            f.write(f'efsearch: {efs}\n')
        sum_recall = 0
        sum_ndc = 0
        sum_nhops = 0
        idx = 0
        sum_fp_rknn = 0
        sum_tp_rknn = 0
        sum_fn_rknn = 0
        sum_tp_indegree = 0
        sum_fp_indegree = 0
        sum_fn_indegree = 0
        sum_tp_lengths = 0
        sum_fp_lengths = 0
        sum_fn_lengths = 0
        sum_tps = 0
        sum_fps = 0
        sum_fns = 0
        # print()
        for q in Q:
            # if idx != special_q:
            #     idx+=1
            #     continue
            topk, ndc, nhops = search(G, X, q, k,  efs, visited_points, glanced_points)
            gt = GT[idx][:k]
            recall = get_recall(topk, gt)
            
            tps, fps, fns = decompose_topk_to_3sets(np.array(topk), np.array(gt))
            sum_tps += len(tps)
            sum_fps += len(fps)
            sum_fns += len(fns)
            
            # tp_n_rknn, fp_n_rknn, fn_n_rknn = analyze_results_k_occurrence(tps, fps, fns, n_rknn)
            # sum_fp_rknn += fp_n_rknn
            # sum_tp_rknn += tp_n_rknn
            # sum_fn_rknn += fn_n_rknn
            
            # tp_indegree, fp_indegree, fn_indegree = analyze_results_indegree(tps, fps, fns, indegree)
            # sum_tp_indegree += tp_indegree
            # sum_fp_indegree += fp_indegree
            # sum_fn_indegree += fn_indegree
                        
            # tp_lengths, fp_lengths, fn_lengths = analyze_results_norm_length(tps, fps, fns, lengths)
            # sum_tp_lengths += tp_lengths
            # sum_fp_lengths += fp_lengths
            # sum_fn_lengths += fn_lengths
            
            idx += 1
            if idx % 1000 == 0:
                print(idx)
            sum_ndc += ndc
            sum_nhops += nhops
            sum_recall += recall
        # print()
        recall = sum_recall / Q.shape[0] / k
        ndc = sum_ndc / Q.shape[0]
        nhops = sum_nhops / Q.shape[0]
        if f is None:
            print(f'recall: {recall}, ndc: {ndc}, nhops: {nhops}')
            print(f'\ttp_rknn: {sum_tp_rknn / sum_tps}, fp_rknn: {sum_fp_rknn / sum_fps}, fn_rknn: {sum_fn_rknn / sum_fns}, tp-fn: {sum_tp_rknn / sum_tps - sum_fn_rknn / sum_fns}')
            print(f'{sum_tp_rknn / sum_tps:.2f} & {sum_fp_rknn / sum_fps:.2f} & {sum_fn_rknn / sum_fns:.2f} & {sum_tp_rknn / sum_tps - sum_fn_rknn / sum_fns:.2f}')
            print(f'\ttp_indegree: {sum_tp_indegree / sum_tps}, fp_indegree: {sum_fp_indegree / sum_fps}, fn_indegree: {sum_fn_indegree / sum_fns}, tp-fn: {sum_tp_indegree / sum_tps - sum_fn_indegree / sum_fns}')
            print(f'{sum_tp_indegree / sum_tps:.2f} & {sum_fp_indegree / sum_fps:.2f} & {sum_fn_indegree / sum_fns:.2f} & {sum_tp_indegree / sum_tps - sum_fn_indegree / sum_fns:.2f}')
            print(f'\ttp_lengths: {sum_tp_lengths / sum_tps}, fp_lengths: {sum_fp_lengths / sum_fps}, fn_lengths: {sum_fn_lengths / sum_fns}, tp-fn: {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns}')
            print(f'{sum_tp_lengths / sum_tps:.2f} & {sum_fp_lengths / sum_fps:.2f} & {sum_fn_lengths / sum_fns:.2f} & {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns:.2f}')
        else:
            f.write(f'recall: {recall}, ndc: {ndc}, nhops: {nhops}')
            f.write('\n')
            # f.write(f'\ttp_rknn: {sum_tp_rknn / sum_tps}, fp_rknn: {sum_fp_rknn / sum_fps}, fn_rknn: {sum_fn_rknn / sum_fns}, tp-fn: {sum_tp_rknn / sum_tps - sum_fn_rknn / sum_fns}')
            # f.write('\n')
            # # f.write(f'{sum_tp_rknn / sum_tps:.2f} & {sum_fp_rknn / sum_fps:.2f} & {sum_fn_rknn / sum_fns:.2f} & {sum_tp_rknn / sum_tps - sum_fn_rknn / sum_fns:.2f}')
            # f.write(f'\ttp_indegree: {sum_tp_indegree / sum_tps}, fp_indegree: {sum_fp_indegree / sum_fps}, fn_indegree: {sum_fn_indegree / sum_fns}, tp-fn: {sum_tp_indegree / sum_tps - sum_fn_indegree / sum_fns}')
            # f.write('\n')
            # # f.write(f'{sum_tp_indegree / sum_tps:.2f} & {sum_fp_indegree / sum_fps:.2f} & {sum_fn_indegree / sum_fns:.2f} & {sum_tp_indegree / sum_tps - sum_fn_indegree / sum_fns:.2f}')
            # f.write(f'\ttp_lengths: {sum_tp_lengths / sum_tps}, fp_lengths: {sum_fp_lengths / sum_fps}, fn_lengths: {sum_fn_lengths / sum_fns}, tp-fn: {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns}')
            # f.write('\n')
            # f.write(f'{sum_tp_lengths / sum_tps:.2f} & {sum_fp_lengths / sum_fps:.2f} & {sum_fn_lengths / sum_fns:.2f} & {sum_tp_lengths / sum_tps - sum_fn_lengths / sum_fns:.2f}')
    if f is not None:
        f.close()
        # write_fbin_simple(f'./figures/{dataset}-hnsw-visit-hub-correlation-ef{efs}.fbin', visited_points)
        # write_fbin_simple(f'./figures/{dataset}-hnsw-glance-hub-correlation-ef{efs}.fbin', glanced_points)
 
def get_recall(topk, groundtruth):
    cnt = 0
    ids = [topk[i][0] for i in range(len(topk))]
    ids = set(ids)
    assert len(ids) == len(topk)
    for id in ids:
        if id in groundtruth:
            cnt += 1
    return cnt 

def analyze_results_hub(topk, gt, n_rknn):
    # find the rknn for each point in topk
    topk_ids = np.array(list(topk[:, 0]), dtype='int32')
    fp_ids = np.setdiff1d(topk_ids, gt)
    tp_ids = np.intersect1d(topk_ids, gt)
    fn_ids = np.setdiff1d(gt, topk_ids)
    
    fp_n_rknn = n_rknn[fp_ids]
    tp_n_rknn = n_rknn[tp_ids]
    fn_n_rknn = n_rknn[fn_ids]
    
    avg_fp_n_rknn = np.average(fp_n_rknn) if len(fp_n_rknn) > 0 else 0
    avg_tp_n_rknn = np.average(tp_n_rknn) if len(tp_n_rknn) > 0 else 0
    avg_fn_n_rknn = np.average(fn_n_rknn) if len(fn_n_rknn) > 0 else 0
    
    return avg_fp_n_rknn, avg_tp_n_rknn, avg_fn_n_rknn
    
source = './data/'
result_source = './results/'
dataset = 'deep'
idx_postfix = '_plain_1th'
efConstruction = 500
Kbuild = 16
M=16
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    # graph_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_K{Kbuild}.nsw.index')
    index_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}.index{idx_postfix}')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs')
    GT_dist_path = os.path.join(source, dataset, f'{dataset}_groundtruth_dist.fvecs')
    KG = 32
    ind_path = os.path.join(source, dataset, f'{dataset}_ind_{KG}.ibin')
    hnsw_ind_path = os.path.join(source, dataset, f'{dataset}_hnsw_ef{efConstruction}_M{M}_ind_{KG}.ibin')
    kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')
    result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_.log{idx_postfix}_unidepth-level')
    
    # recalls = []
    # with open(result_path, 'r') as f:
    #     lines = f.readlines()
    #     for i in range(3, len(lines), 7):
    #         line = lines[i].strip().split(',')[:-1]
    #         recalls.append(np.array([int(x) for x in line]))

    # low_recall_positions = []
    # for recall in recalls:
    #     low_recall_positions.append(np.where(recall < 40)[0])
    X = read_fvecs(base_path)
    # G = read_hnsw_index_aligned(index_path, X.shape[1])
    # G = read_hnsw_index_unaligned(index_path, X.shape[1])
    Q = read_fvecs(query_path)
    # Q = read_fvecs(query_path)
    GT = read_ivecs(GT_path)
    GT_dist = read_fvecs(GT_dist_path)
    KGraph = read_ivecs(kgraph_path)
    lengths = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'))
    # q_lengths = get_query_length(X, Q)
    n_rknn = read_ibin_simple(ind_path)
    # indegree = get_indegree_list(G)
    # write_ibin_simple(hnsw_ind_path, indegree)
    indegree = read_ibin_simple(hnsw_ind_path)
    # revG = get_reversed_graph_list(G)
    
    # # plot the hist of gt_dist[0]
    # plt.hist(GT_dist[:, 0], bins=50, edgecolor='black', linewidth=1.2)
    # plt.xlabel('1NN distance')
    # plt.ylabel('number of queries')
    # plt.title(f'{dataset} query to 1NN distance distribution', fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-to-1NN-distance-distribution.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-query-to-1NN-distance-distribution.png')
    
    # q_k_occurs = []
    # for j in range(Q.shape[0]):
    #     gt = GT[j][:50]
    #     sum_k_occur = 0
    #     for j in range(50):
    #         sum_k_occur += n_rknn[gt[j]]
    #     q_k_occurs.append(sum_k_occur / 50)
    # q_k_occurs = np.array(q_k_occurs)
    # # print(np.average(q_k_occurs))
    # # find the indexes of 200 minimum k_occurs
    # k_occurs_indexes = np.argsort(q_k_occurs)
    
    # q_knn_dist_range = []
    # for i in range(Q.shape[0]):
    #     gt = GT_dist[i]
    #     q_knn_dist_range.append(GT_dist[i][-1] - GT_dist[i][0])
    # q_knn_dist_range = np.array(q_knn_dist_range)
    # knn_dist_range_indexes = np.argsort(q_knn_dist_range)

    
    
    q_lids = get_lids(GT_dist[:, :50], 50)
    # find the indexes of 200 minimum k_occurs
    lids_indexes = np.argsort(q_lids)
    npart = 10
    
    
    # compact_q_indexes = []
    # high_k_occur_q_indexes = []
    # contrastive_q_indexes = []
    # k_occurs = []
    # for i in range(10000):
    #     q_id = lids_indexes[i]
    #     lid = q_lids[q_id]
    #     gt_dist_min = GT_dist[q_id][0]
    #     gt_dist_max = GT_dist[q_id][-1]
    #     # k_occurs.append(q_k_occurs[q_id])]
    #     # if q_k_occurs[q_id] > 60:
    #     #     compact_q_indexes.append(q_id)
    #     # elif len(contrastive_q_indexes) < len(compact_q_indexes):
    #     #     contrastive_q_indexes.append(q_id)
    #     if 0.75 < gt_dist_min :
    #         # k_occurs.append(q_k_occurs[q_id])
    #         compact_q_indexes.append(q_id)
    #         # if q_k_occurs[q_id] > 25:
    #             # high_k_occur_q_indexes.append(q_id)
    #     # elif len(contrastive_q_indexes) < len(high_k_occur_q_indexes):
    #     #     contrastive_q_indexes.append(q_id)
    #         # compact_q_indexes.append(q_id)
            
    #         # print(f'{i}: q_id: {q_id}, lid: {lid}, k_occur:{q_k_occurs[q_id]}, gt_dist_min: {gt_dist_min}, gt_dist_max: {gt_dist_max}, gt_dist_range: {gt_dist_max - gt_dist_min}')
    #     # elif len(contrastive_q_indexes) < len(compact_q_indexes):
    #     #     contrastive_q_indexes.append(q_id)
         
    # plot the hist of k_occurs
    # print(len(k_occurs))
    # k_occurs = np.array(k_occurs)
    # print(f'avg k-occur: {np.average(k_occurs)}, min-k-occur:{np.min(k_occurs)}, max k-occur: {np.max(k_occurs)}')
    # plt.hist(k_occurs, bins=50, edgecolor='black', linewidth=1.2)
    # plt.xlabel('k_occurs for large lid, far-from 1NN')
    # plt.ylabel('number of queries')
    # plt.title(f'{dataset} k_occurs for small lid, far-from 1NN', fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-k_occurs-for-large-lid-far-from-1NN.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-k_occurs-for-large-lid-far-from-1NN.png')
    # high_k_occur_queries = Q[high_k_occur_q_indexes]
    # high_k_occur_ground_truth = GT[high_k_occur_q_indexes]
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_high_k_occur'), high_k_occur_queries)
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_high_k_occur'), high_k_occur_ground_truth)
    # print(len(high_k_occur_queries))
    # constrastive_queries = Q[contrastive_q_indexes]
    # constrastive_ground_truth = GT[contrastive_q_indexes]
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_contrastive'), constrastive_queries)
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_contrastive'), constrastive_ground_truth)
    # print(len(contrastive_q_indexes))
        
    # split compact query into 2 parts by k-occurrence
    # print(len(compact_q_indexes))
    # compact_q_indexes = np.array(compact_q_indexes)
    # compact_q_k_occurs = q_knn_dist_range[compact_q_indexes]
    # k_occur_indexes = np.argsort(compact_q_k_occurs)
    
    # compact_low_k_occur_q_indexes = compact_q_indexes[k_occur_indexes[: 50]]
    # compact_high_k_occur_q_indexes = compact_q_indexes[k_occur_indexes[-50:]]
    # compact_low_k_occur_queries = Q[compact_low_k_occur_q_indexes]
    # compact_low_k_occur_ground_truth = GT[compact_low_k_occur_q_indexes]
    # k_occurs = []
    # lids = []
    # knn_dist_range = []
    # dist_1nn = []
    # for i in range(len(compact_low_k_occur_q_indexes)):
    #     q_id = compact_low_k_occur_q_indexes[i]
    #     k_occurs.append(q_k_occurs[q_id])
    #     lids.append(q_lids[q_id])
    #     dist_1nn.append(GT_dist[q_id][0])
    #     knn_dist_range.append(q_knn_dist_range[q_id])
    # k_occurs = np.array(k_occurs)
    # lids = np.array(lids)
    # dist_1nn = np.array(dist_1nn)
    # print(f'low dist range: min dist range: {np.min(knn_dist_range)}, max dist range: {np.max(knn_dist_range)}, avg dist range: {np.average(knn_dist_range)}, avg 1nn dist: {np.average(dist_1nn)}')
    # # print(f'low 1nn dist: min 1nn dist: {np.min(dist_1nn)}, max 1nn dist: {np.max(dist_1nn)}, avg 1nn dist: {np.average(dist_1nn)}')
    # # print(f'low k_occur: avg k-occur: {np.average(k_occurs)}, min-k-occur:{np.min(k_occurs)}, max k-occur: {np.max(k_occurs)}, avg lid: {np.average(lids)}')
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_compact_low_k_occur'), compact_low_k_occur_ground_truth)
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_compact_low_k_occur'), compact_low_k_occur_queries)
    # print(len(compact_low_k_occur_queries))
    # compact_high_k_occur_queries = Q[compact_high_k_occur_q_indexes]
    # compact_high_k_occur_ground_truth = GT[compact_high_k_occur_q_indexes]
    # k_occurs = []
    # lids = []
    # dist_1nn = []
    # knn_dist_range = []
    # for i in range(len(compact_high_k_occur_q_indexes)):
    #     q_id = compact_high_k_occur_q_indexes[i]
    #     k_occurs.append(q_k_occurs[q_id])
    #     lids.append(q_lids[q_id])
    #     dist_1nn.append(GT_dist[q_id][0])
    #     knn_dist_range.append(q_knn_dist_range[q_id])
    # k_occurs = np.array(k_occurs)
    # lids = np.array(lids)
    # print(f'low dist range: min dist range: {np.min(knn_dist_range)}, max dist range: {np.max(knn_dist_range)}, avg dist range: {np.average(knn_dist_range)}, avg 1nn dist: {np.average(dist_1nn)}')
    # # print(f'low 1nn dist: min 1nn dist: {np.min(dist_1nn)}, max 1nn dist: {np.max(dist_1nn)}, avg 1nn dist: {np.average(dist_1nn)}')
    # # print(f'high k_occur: avg k-occur: {np.average(k_occurs)}, min k-occur:{np.min(k_occurs)}, max k-occur: {np.max(k_occurs)}, avg lid: {np.average(lids)}')
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_compact_high_k_occur'), compact_high_k_occur_ground_truth)
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_compact_high_k_occur'), compact_high_k_occur_queries)
    # print(len(compact_high_k_occur_queries))
    
    
    # compact_queries = Q[compact_q_indexes]
    # compact_ground_truth = GT[compact_q_indexes]
    # k_occurs = []
    # lids = []
    # for i in range(len(compact_q_indexes)):
    #     q_id = compact_q_indexes[i]
    #     k_occurs.append(n_rknn[q_id])
    #     lids.append(q_lids[q_id])
    # k_occurs = np.array(k_occurs)
    # lids = np.array(lids)
    # print(f'low k_occur: avg k-occur: {np.average(k_occurs)}, min-k-occur:{np.min(k_occurs)}, max k-occur: {np.max(k_occurs)}, avg lid: {np.average(lids)}')
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_compact'), compact_ground_truth)
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_compact'), compact_queries)
    # print(len(compact_queries))
    
    # contrastive_queries = Q[contrastive_q_indexes]
    # contrastive_ground_truth = GT[contrastive_q_indexes]
    # k_occurs = []
    # lids = []
    # for i in range(len(contrastive_q_indexes)):
    #     q_id = contrastive_q_indexes[i]
    #     k_occurs.append(n_rknn[q_id])
    #     lids.append(q_lids[q_id])
    # k_occurs = np.array(k_occurs)
    # lids = np.array(lids)
    # print(f'low k_occur: avg k-occur: {np.average(k_occurs)}, min-k-occur:{np.min(k_occurs)}, max k-occur: {np.max(k_occurs)}, avg lid: {np.average(lids)}')

    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_contrastive'), contrastive_ground_truth)
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_contrastive'), contrastive_queries)
    # print(len(contrastive_q_indexes))
    
    # lid_hard_3k_queries = Q[lids_indexes[-3000:]]
    # lid_hard_3k_ground_truth = GT[lids_indexes[-3000:]]
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_lid_hard_3k'), lid_hard_3k_ground_truth)
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_lid_hard_3k'), lid_hard_3k_queries)
    
    # plt.hist(q_lids, bins=100)
    # plt.xlabel('k_occurs')
    # plt.ylabel('number of queries')
    # plt.title(f'{dataset} k_occurs distribution')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k_occurs-distribution.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-query-k_occurs-distribution.png')
    
    # query_splits = []
    # q_index_splits = []
    # r = q_lids.max() - q_lids.min()
    # part_size = r / npart
    # start = 0
    # for t in range(npart):
    #     part_max = q_lids.min() + (t+1) * part_size
    #     if t == npart - 1:
    #         part_max = q_lids.max() + 1
    #     print(f'from {q_lids.min() + t * part_size} to {part_max}')
    #     pre_start = start
    #     while start < Q.shape[0] and q_lids[lids_indexes[start]] < part_max:
    #         start += 1
    #     qindexes = lids_indexes[pre_start: start]
    #     q_index_splits.append(qindexes)
    #     query_splits.append(Q[qindexes])
    #     qpath = os.path.join(source, dataset, f'{dataset}_query.fvecs_lid_unidepth-level{t}')
    #     gtpath = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_lid_unidepth-level{t}')
    #     write_fvecs(qpath, Q[qindexes])
    #     write_ivecs(gtpath, GT[qindexes])

    
    # part_size = Q.shape[0] // npart
    # query_splits = []
    # for i in range(npart):
    #     qindexes = lids_indexes[i*part_size:(i+1)*part_size]
    #     query_splits.append(Q[qindexes])
    #     qpath = os.path.join(source, dataset, f'{dataset}_query.fvecs_lid_level{i}')
    #     gtpath = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_lid_level{i}')
    #     write_fvecs(qpath, Q[qindexes])
    #     write_ivecs(gtpath, GT[qindexes])
    
    # find the indexes of 200 maximum k_occurs
    # max_q_k_occurs_indexes = np.argsort(q_lids)[::-1][:200]
    
    
    # hard_q_lengths = q_lengths[min_q_k_occurs_indexes]
    # easy_q_lengths = q_lengths[max_q_k_occurs_indexes]
    # print(np.average(hard_q_lengths), np.average(easy_q_lengths))
    
    # hard_queries = Q[min_q_k_occurs_indexes]
    # hard_ground_truth = GT[min_q_k_occurs_indexes]
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_hard'), hard_ground_truth)
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_hard'), hard_queries)
    # easy_queries = Q[max_q_k_occurs_indexes]
    # easy_ground_truth = GT[max_q_k_occurs_indexes]
    # write_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs_easy'), easy_queries)
    # write_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs_easy'), easy_ground_truth)
    
        
        


    