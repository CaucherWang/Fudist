import os 
import numpy as np
datsets_map = {
    # 'imagenet': (6, 200),
    # 'msong': (6, 1000),
    # 'word2vec': (6, 1000),
    # 'ukbench': (8, 200),
    'deep': (8, 1000, 16),
    'gist': (8, 1000, 96),
    'glove1.2m': (8, 1000, 20),
    'sift': (8, 1000, 16),
    # 'tiny5m': (8, 1000),
}

def read_floats(filename, c_contiguous=True):
    print(f"Reading File - {filename}")
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    if c_contiguous:
        fv = fv.copy()
    return fv


ratios = {}
source = './data/'
dataset = 'deep'
path = os.path.join(source, dataset)
paa_path = os.path.join(path, f'PAA_{datsets_map[dataset][2]}_approx_dist.floats')
pq_path = os.path.join(path, f'PQ_{datsets_map[dataset][0]}_256_approx_dist.floats')
opq_path = os.path.join(path, f'OPQ_{datsets_map[dataset][0]}_256_approx_dist.floats')
lsh_path = os.path.join(path, f'LSH_64_approx_dist.floats')
svd_path = os.path.join(path, f'SVD_0.8_approx_dist.floats')
ads_path = os.path.join(path, f'ADS_0.8_approx_dist.floats')
ads2_path = os.path.join(path, f'ADS_0.5_approx_dist.floats')

ads = read_floats(ads_path)
ads2 = read_floats(ads2_path)
print(np.mean(ads))
print(np.mean(ads2))
