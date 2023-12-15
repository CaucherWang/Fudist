from utils import *

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,4.8))


source = './data/'
result_source = './results/'
exp = 'kgraph'
dataset = 'deep'
idx_postfix = '_plain'
Kbuild =475
kgraph_od = '_500'
M=100
efConstruction = 2000
R = 32
L = 40
C = 500
if __name__ == "__main__":
    standard_hnsw_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}_hnsw.ibin{idx_postfix}')
    standard_reversed_hnsw_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}_hnsw.ibin{idx_postfix}_reversed')
    nsg_path = os.path.join(source, dataset, f'{dataset}_L{L}_R{R}_C{C}.nsg')
    reversed_nsg_path = os.path.join(source, dataset, f'{dataset}_L{L}_R{R}_C{C}.nsg_reversed')
    kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth{kgraph_od}.ivecs')
    reversed_kgraph_path = os.path.join(source, dataset, f'{dataset}_K{Kbuild}_self_groundtruth.ivecs_reversed')

    if exp == 'kgraph':  
        # KGraph = read_ivecs(kgraph_path)
        # KGraph_clean = clean_kgraph(KGraph)
        # write_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth{kgraph_od}.ivecs_clean'), KGraph_clean)

        KGraph = read_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth{kgraph_od}.ivecs_clean'))[:, :Kbuild]
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
        revG = get_reversed_graph_list(KGraph)
        write_obj(reversed_kgraph_path, revG)
        new_path = reversed_kgraph_path + '_std'
        transform_kgraph2std(new_path, revG)
        # revG = read_obj(reversed_kgraph_path)
    elif exp == 'hnsw':
        hnsw = read_ibin(standard_hnsw_path)
        revG = get_reverse_graph(hnsw)
        write_obj(standard_reversed_hnsw_path, revG)
        revG = read_obj(standard_reversed_hnsw_path)
    elif exp == 'nsg':
        ep, nsg = read_nsg(nsg_path)
        revG = get_reversed_graph_list(nsg)
        write_obj(reversed_nsg_path, revG)
        revG = read_obj(reversed_nsg_path)
    else:
        print(exp)
        raise NotImplementedError
