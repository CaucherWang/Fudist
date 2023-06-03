# encoding=utf-8
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os
# fp = fm.FontProperties(family='Latin Modern Math')

fig = plt.figure(figsize=(6,4.2))

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
            if(line[0] == '|||'):
                continue
            if(len_line < 30):
                recall.clear()
                time.clear()
                continue

            recall.append(float(line[1]))
            time.append(float(line[2]) / 1000.0)
        return recall, time
    

hnsw = {}
ads = {}
pca = {}
pq = {}
opq = {}
lsh = {}
finger = {}

datsets_map = {
    'imagenet': (16,6, 16),
    'msong': (8,6),
    'word2vec': (16,6),
    'ukbench': (16,8),
    'deep': (16, 8, 16),
    'gist': (16, 8, 96),
    'glove1.2m': (128, 8, 20),
    'sift': (16, 8, 16),
    'tiny5m': (16, 8),
    'uqv':(16, 8, 16),
    'glove-100':(16, 4, 16),
    'crawl': (16, 6, 16),
    'enron':(16, 8, 64),
    'mnist':(8, 8, 64),
    'cifar': (8,8,64),
    'sun': (8, 8, 64),
    'trevi':(16, 8, 64),
    'notre':(8, 8, 16),
    'nuswide':(48, 10, 64),
}

source = './results/'
dataset = 'nuswide'
ef = 500
M = datsets_map[dataset][0]
pq_m = datsets_map[dataset][1]
pq_ks = 256
lsh_lowdim = datsets_map[dataset][2]
path = os.path.join(source, dataset)
hnsw_path = os.path.join(path, f'{dataset}_ef{ef}_M{M}_.log')
ads_path = os.path.join(path, f'{dataset}_ef{ef}_M{M}_ADS.log')
lsh_path = os.path.join(path, f'{dataset}_ef{ef}_M{M}_LSH{lsh_lowdim}.log')
pca_path = os.path.join(path, f'{dataset}_ef{ef}_M{M}_PCA.log')
pq_path = os.path.join(path, f'{dataset}_ef{ef}_M{M}_PQ{pq_m}-{pq_ks}.log')
opq_path = os.path.join(path, f'{dataset}_ef{ef}_M{M}_OPQ{pq_m}-{pq_ks}.log')
# finger_path

hnsw = resolve_log(hnsw_path)
ads = resolve_log(ads_path)
lsh = resolve_log(lsh_path)
pca = resolve_log(pca_path)
# pq = resolve_log(pq_path)
opq = resolve_log(opq_path)

k = 20
width = 1.4
# gist
# plt.plot(hnsw[1][2:-2], hnsw[0][2:-2], marker='d', label='HNSW', markersize=5, linewidth=width, color='mediumpurple', alpha=0.9)
# plt.plot(ads[1][2:], ads[0][2:], marker='o', label='ADS', markersize=4, linewidth=width, color='firebrick', alpha=1)
# plt.plot(pca[1][2:], pca[0][2:],  marker='*', label='PCA', markersize=7, linewidth=width, color='indianred', alpha=0.9)
# plt.plot(lsh[1][2:-1], lsh[0][2:-1], marker='+', label='LSH', markersize=7, linewidth=width, color='olive')
# plt.plot(pq[1], pq[0], marker='x', label='PQ', markersize=8, linewidth=width, color='darkgray')
# plt.plot(opq[1][2:], opq[0][2:], marker='D', label='OPQ', markersize=3, linewidth=width, color='steelblue', alpha=0.9)


# deep
plt.plot(hnsw[1][:], hnsw[0][:], marker='d', label='HNSW', markersize=5, linewidth=width, color='mediumpurple', alpha=0.9)
plt.plot(ads[1][:], ads[0][:], marker='o', label='ADS', markersize=4, linewidth=width, color='firebrick', alpha=1)
plt.plot(pca[1][:], pca[0][:],  marker='*', label='PCA', markersize=7, linewidth=width, color='indianred', alpha=0.9)
plt.plot(lsh[1][:], lsh[0][:], marker='+', label='LSH', markersize=7, linewidth=width, color='olive')
# plt.plot(pq[1], pq[0], marker='x', label='PQ', markersize=4, linewidth=width, color='darkgray')
plt.plot(opq[1][:], opq[0][:], marker='D', label='OPQ', markersize=3, linewidth=width, color='steelblue', alpha=0.9)


# plt.xlim(0.1,1)
# plt.ylim(1, 10000)

# plt.yscale('log')
# plt.xscale('log')
# plt.xticks([0.1,0.3,0.5,0.7,1],[0.1,0.3,0.5,0.7,1],fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.yticks([1e1,1e2,1e3,1e4],['$10^1$','$10^2$','$10^3$','$10^4$'],fontsize=18)
# plt.yticks([1e1,1e2,1e3,1e4],['$10^1$','$10^2$','$10^3$','$10^4$'],fontsize=18)

# plt.legend(loc="best", fontsize=15)
plt.ylabel(f"Recall{k}@{k} (%)", fontsize=16)
plt.xlabel("Query time (ms)", fontsize=16)
# plt.title('rand-256-100m (100GB)', fontsize=20)
plt.legend(loc='best', fontsize=16)  #显示图中左上角的标识区域
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=6,mode="expand", borderaxespad=0.,fontsize=15.5)  #显示图中左上角的标识区域

plt.tight_layout()
plt.savefig(f'./figures/fig/ann-{dataset}.png',  format='png')

# plt.show()
# plt.savefig('../figs/approx-node-full-%s-recall.png' % ds,  format='png')
# plt.savefig('../figs/approx-full-title.png', format='png')
