# encoding=utf-8
import matplotlib.pyplot as plt
from math import *
import numpy as np
import os
# plt.figure(figsize=(20, 4.7))
plt.figure(figsize=(6.5,4))

def read_floats(filename, c_contiguous=True):
    print(f"Reading File - {filename}")
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    if c_contiguous:
        fv = fv.copy()
    return fv


datsets_map = {
    # 'imagenet': (6, 200),
    # 'msong': (6, 1000),
    # 'word2vec': (6, 1000),
    # 'ukbench': (8, 200),
    'deep': (8, 1000, 16),
    'gist': (8, 1000, 96),
    'glove1.2m': (8, 1000, 20),
    'sift': (8, 1000, 16),
    'tiny5m': (8, 1000),
}

# for dataset in datsets_map:
#     source = './data/'
#     path = os.path.join(source, dataset)
#     result_path = os.path.join(path, f'OPQ_{datsets_map[dataset][0]}_256_approx_dist.floats')
#     ratios[dataset] = list(read_floats(result_path))
# plot the shaded box plots on ratios, the x-axis is the dataset name, the y-axis is the ratio, and make the box plots more beautiful
# plt.boxplot(ratios.values(), labels=ratios.keys(), showfliers=False, medianprops={'color':'blue', 'linewidth':2}, boxprops={'color':'black', 'linewidth':2}, whiskerprops={'color':'black', 'linewidth':2}, capprops={'color':'black', 'linewidth':2})
# plt.savefig('./figures/fig/app-ratio-opq.png',  format='png')



ratios = {}
source = './data/'
dataset = 'gist'
path = os.path.join(source, dataset)
for sample in [0.2, 0.4, 0.6, 0.8]:
    ratios[f'PCA-{sample}'] = list(read_floats(os.path.join(path, f'PCA_{sample}_approx_dist.floats')))

for sample in [0.2, 0.4, 0.6, 0.8]:
    ratios[f'DWT-{sample}'] = list(read_floats(os.path.join(path, f'DWT_{sample}_approx_dist.floats')))

for sample in [0.2, 0.4, 0.6, 0.8]:
    ratios[f'ADS-{sample}'] = list(read_floats(os.path.join(path, f'ADS_{sample}_approx_dist.floats')))



# plot the shaded box plots on ratios, the x-axis is the method name, the y-axis is the ratio
plt.boxplot(ratios.values(), labels=ratios.keys(), showfliers=False, medianprops={'color':'blue', 'linewidth':2}, boxprops={'color':'black', 'linewidth':2}, whiskerprops={'color':'black', 'linewidth':2}, capprops={'color':'black', 'linewidth':2})
plt.ylabel("Approximation Ratio", fontsize=16)
plt.tight_layout()

plt.savefig(f'./figures/fig/app-ratio-incremental-{dataset}.png',  format='png')
# x = ['1k', '2.5k', '5k', '10k', '25k', '50k','100k']
# x = [1000, 2500, 5000, 10000, 25000, 50000, 100000]
# paris = [27.62, 70.9, 197.6, 503.37, 1208, 2442]


# Dumpy = [41.68, 100.08, 198.17, 397.18, 987.92, 1980.83]


# ds = 'rand'
# width = 2.0
# # plt.plot(x[:-1], fuzzy30[ds][:-1], marker='^', label='Fuzzy-30', markersize=10, linewidth=width, color='darkred')

# plt.plot(x[:-1], paris, marker='x', label='PARIS+', markersize=11, linewidth=width, color='darkgray')
# plt.plot(x[:-1], Dumpy, marker='*', label='DumpyOS', markersize=13, linewidth=width, color='indianred', alpha=0.9)

# def forward(x):
#     return 2 ** x

# def inverse(x):
#     return log2(x)
# # plt.legend(ncol=2,loc='best',  borderaxespad=0.,fontsize=14)  #显示图中左上角的标识区域
# # plt.xscale('function', functions=(forward, inverse))
# plt.xscale('log')
# plt.xticks(x, fontsize=16)
# # plt.xticks([1000, 10000, 25000, 50000, 100000],['1k','10k','25k', '50k','100k'],fontsize=16)
# # plt.yscale('log')
# plt.yticks([0, 1000, 2000, 3000, 4000, 5000],fontsize=16)
# plt.legend(loc="upper left" , fontsize = 18)
# plt.xlabel("#queries", fontsize=16)
# plt.ylabel("Accumulated query time (min)", fontsize=14)
# # plt.title('rand-256-100m, 200 hard queries')
# plt.tight_layout()
# # plt.show()
# plt.savefig('./figures/approx-node-full-%s-recall.png' % ds,  format='png')
# plt.savefig('../figs/approx-full-title.png', format='png')
