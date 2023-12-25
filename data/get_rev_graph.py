from utils import *

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,4.8))


source = './data/'
result_source = './results/'
exp = 'ssg'
dataset = 'sift'
idx_postfix = '_plain'
Kbuild = 475
kgraph_od = '_10000'
KMRNG = 2047
M=50
efConstruction = 500
R = 32
L = 40
C = 500
if __name__ == "__main__":
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    nsg_path = os.path.join(source, dataset, f'{dataset}_L{L}_R{R}_C{C}.nsg')
    reversed_nsg_path = os.path.join(source, dataset, f'{dataset}_L{L}_R{R}_C{C}.nsg_reversed')
    kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth{kgraph_od}.ivecs')
    reversed_kgraph_path = os.path.join(source, dataset, f'{dataset}_K{Kbuild}_self_groundtruth.ivecs_reversed')

    if exp == 'kgraph':  
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
        # new_path = reversed_kgraph_path + '_std'
        # transform_kgraph2std(new_path, revG)
        # revG = read_obj(reversed_kgraph_path)
    elif exp == 'hnsw':
        efConstruction = 2000
        M = 100
        index_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}.index_plain')
        standard_hnsw_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}_hnsw.ibin{idx_postfix}')
        standard_reversed_hnsw_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}_hnsw.ibin{idx_postfix}_reversed')

        Q = read_fvecs(query_path)
        hnsw = read_hnsw_index_aligned(index_path, Q.shape[1])
        new_hnsw = shuffled_hnsw_to_standard_form(hnsw, np.arange(len(hnsw)), M)
        write_ibin(standard_hnsw_path, new_hnsw)

        revG = get_reversed_graph_list(hnsw)
        write_obj(standard_reversed_hnsw_path, revG)
        transform_kgraph2std(standard_reversed_hnsw_path + '_std', revG)
    elif exp == 'nsg':
        ep, nsg = read_nsg(nsg_path)
        revG = get_reversed_graph_list(nsg)
        write_obj(reversed_nsg_path, revG)
        revG = read_obj(reversed_nsg_path)
    elif exp == 'mrng':
        mrng_path = os.path.join(source, dataset, f'{dataset}_K{KMRNG}.mrng')
        reversed_mrng_path = os.path.join(source, dataset, f'{dataset}_K{KMRNG}.mrng_reversed')
        mrng = read_mrng(mrng_path)
        revG = get_reversed_graph_list(mrng)
        write_obj(reversed_mrng_path, revG)
        # revG = read_obj(reversed_mrng_path)
        new_path = reversed_mrng_path + '_std'
        transform_kgraph2std(new_path, revG)
    elif exp == 'ssg':
        alpha = 60
        ssg_path = os.path.join(source, dataset, f'{dataset}_K{KMRNG}_alpha{alpha}.ssg')
        reversed_ssg_path = os.path.join(source, dataset, f'{dataset}_K{KMRNG}_alpha{alpha}.ssg_reversed')
        ssg = read_mrng(ssg_path)
        revG = get_reversed_graph_list(ssg)
        write_obj(reversed_ssg_path, revG)
        # revG = read_obj(reversed_mrng_path)
        new_path = reversed_ssg_path + '_std'
        transform_kgraph2std(new_path, revG)
    elif exp == 'taumg':
        ssg_path = os.path.join(source, dataset, f'{dataset}_K{KMRNG}.taumg')
        reversed_ssg_path = os.path.join(source, dataset, f'{dataset}_K{KMRNG}.taumg_reversed')
        ssg = read_mrng(ssg_path)
        revG = get_reversed_graph_list(ssg)
        write_obj(reversed_ssg_path, revG)
        # revG = read_obj(reversed_mrng_path)
        new_path = reversed_ssg_path + '_std'
        transform_kgraph2std(new_path, revG)
    else:
        print(exp)
        raise NotImplementedError
