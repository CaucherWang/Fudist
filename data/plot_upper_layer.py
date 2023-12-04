# encoding=utf-8
import os
import numpy as np
from pprint import pprint
from utils import *

fig = plt.figure(figsize=(6,5))

def resolve_log(file_path):
    rset = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        recall = []
        time = []
        for line in lines:
            line = line.strip()
            len_line = len(line)
            line = line.split(' ')
            if(line[0] == '|||' or line[0] == 'SEARCH'):
                continue
            if(len_line < 30):
                recall.clear()
                time.clear()
                continue

            recall.append(float(line[1]))
            time.append(float(line[19]))
        return recall, time
    
    
datsets_map = {
    # 'cifar': (8,8,64),
    # 'crawl': (16, 6, 16),
    # 'deep': (16, 8, 16),
    # 'enron':(16, 8, 64),
    # 'gist': (16, 8, 64),
    # 'glove1.2m': (128, 8, 20),
    # 'glove-100':(16, 4, 16),
    # 'imagenet': (16,6, 64),
    # 'mnist':(8, 8, 64),
    # 'msong': (8, 6, 64),
    # 'notre':(8, 8, 16),
    # 'nuswide':(48, 10, 64),
    # 'sift': (16, 8, 16),
    # 'sun': (8, 8, 64),
    # 'tiny5m': (48, 8, 16),
    # 'trevi':(16, 8, 64),
    # 'ukbench': (16,8,16),
    # 'uqv':(16, 8, 16),
    # 'word2vec': (48, 6, 16),
}

source = './results/'
dataset = 'msturing100m'
ef = 500
# M = datsets_map[dataset][0]
M = 16
path = os.path.join(source, dataset)
hnsw_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log')
plain_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain')

k = 20
width = 2
hnsw = resolve_log(hnsw_path)
plain = resolve_log(plain_path)

# plt.plot(hnsw[1], hnsw[0], **plot_info[0], label='HNSW', markersize=20)
# plt.plot(plain[1], plain[0], **plot_info[1], label='HNSW$_0$', markersize=20, color='steelblue')

plt.plot(hnsw[1], hnsw[0], marker='o', markersize=12, linewidth=width, label='HNSW', color='steelblue' )
plt.plot(plain[1], plain[0],marker='*', markersize=16,linewidth=width, label='HNSW$_0$', color='indianred')



# plt.legend(loc="best", fontsize=15)
plt.xticks()
plt.yticks()
plt.xlabel("NDC")
plt.ylabel(r"Recall@20")

plt.legend(loc='best')

# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=6,mode="expand", borderaxespad=0.)  #显示图中左上角的标识区域
# plt.title('rand-256-100m (100GB)', fontsize=20)
# plt.legend(loc='best',ncol = 2, fontsize=16)  #显示图中左上角的标识区域
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=6,mode="expand", borderaxespad=0.,fontsize=15.5)  #显示图中左上角的标识区域

plt.tight_layout()
# plt.savefig(f'./figures/bulletin-{dataset}-dist2mean.png',  format='png')
# plt.savefig(f'./figures/{dataset}-outlink-density.png',  format='png')
# check the path exists and if not create it
if not os.path.exists(f'./figures/{dataset}'):
    os.makedirs(f'./figures/{dataset}')
print(f'save to ./figures/{dataset}/{dataset}-upper-layer.png')
plt.savefig(f'./figures/{dataset}/{dataset}-upper-layer.png',  format='png')
# plt.show()
# plt.savefig('../figs/approx-node-full-%s-recall.png' % ds,  format='png')
# plt.savefig('../figs/approx-full-title.png', format='png')
