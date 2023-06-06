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
        dist = []
        for line in lines:
            line = line.strip()
            len_line = len(line)
            line = line.split(' ')
            if(line[0] == '|||'):
                continue
            if(len_line < 30):
                recall.clear()
                dist.clear()
                continue

            recall.append(float(line[1]))
            dist.append(float(line[-1]) / 1000)
        return [recall, dist]
    

hnsw = {}
ads = {}
pca = {}
pq = {}
opq = {}
lsh = {}
finger = {}

datsets_map = {
    'cifar': (8,8,64),
    'crawl': (16, 6, 16),
    'deep': (16, 8, 16),
    'enron':(16, 8, 64),
    'gist': (16, 8, 64),
    'glove1.2m': (128, 8, 20),
    'glove-100':(16, 4, 16),
    'imagenet': (16,6, 16),
    'mnist':(8, 8, 64),
    'msong': (8,6),
    'notre':(8, 8, 16),
    'nuswide':(48, 10, 64),
    'sift': (16, 8, 16),
    'sun': (8, 8, 64),
    'tiny5m': (48, 8, 16),
    'trevi':(16, 8, 64),
    'ukbench': (16,8),
    'uqv':(16, 8, 16),
    'word2vec': (48, 6, 16),
}

source = './results/'
dataset = 'gist'
ef = 500
M = datsets_map[dataset][0]
pq_m = datsets_map[dataset][1]
pq_ks = 256
lsh_lowdim = datsets_map[dataset][2]
path = os.path.join(source, dataset)
hnsw_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log')
ads_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_ADS.log')
lsh_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_LSH{lsh_lowdim}.log')
pca_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_PCA.log')
pq_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_PQ{pq_m}-{pq_ks}.log')
opq_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_OPQ{pq_m}-{pq_ks}.log')
dwt_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_DWT.log')
finger_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_FINGER.log')

hnsw = resolve_log(hnsw_path)
ads = resolve_log(ads_path)
lsh = resolve_log(lsh_path)
pca = resolve_log(pca_path)
# pq = resolve_log(pq_path)
opq = resolve_log(opq_path)
dwt = resolve_log(dwt_path)
finger = resolve_log(finger_path)
# copy pca[1] to hnsw[1]
hnsw[1] = pca[1]
methods = [ads, pca, lsh, opq, dwt, hnsw]

# fit a curve using hnsw[1] and hnsw[0]


for method in methods:
    for i in range(len(method[0])):
        method[0][i] /= hnsw[0][i]


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
plt.plot(dwt[1], dwt[0], marker='x', label='DWT', markersize=4, linewidth=width, color='darkgray')
plt.plot(opq[1][:], opq[0][:], marker='D', label='OPQ', markersize=3, linewidth=width, color='steelblue', alpha=0.9)
# plt.plot(finger[1][:], finger[0][:], marker='s', label='FINGER', markersize=3, linewidth=width, color='darkorange', alpha=0.9)

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
plt.ylabel(f"Relative accuracy", fontsize=16)
plt.xlabel("# Distance comparison ($10^3$)", fontsize=16)
# plt.title('rand-256-100m (100GB)', fontsize=20)
plt.legend(loc='best', fontsize=16)  #显示图中左上角的标识区域
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=6,mode="expand", borderaxespad=0.,fontsize=15.5)  #显示图中左上角的标识区域

plt.tight_layout()
plt.savefig(f'./figures/fig/acc-loss-{dataset}.png',  format='png')

# plt.show()
# plt.savefig('../figs/approx-node-full-%s-recall.png' % ds,  format='png')
# plt.savefig('../figs/approx-full-title.png', format='png')