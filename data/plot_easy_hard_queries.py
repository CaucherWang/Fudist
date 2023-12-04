# encoding=utf-8
import os
import numpy as np
from pprint import pprint
from utils import *

fig = plt.figure(figsize=(8,7))

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
            if(line[0] == '|||' or line[0] == 'SEARCH' or len_line > 200 or line[0][-1] == ','):
                continue
            if(len_line < 30):
                recall.clear()
                time.clear()
                continue

            recall.append(float(line[1]))
            time.append(float(line[19]))
        return recall, time
    
def resolve_python_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        recall = []
        time = []
        for line in lines:
            if line.startswith('efsearch'):
                line = line.strip()
                line = line.split(' ')
                # remove all empty elements in line
                line = [x for x in line if x != '']
                recall.append(float(line[3][:-1]) * 100)
                time.append(float(line[5][:-1]))
            # continue if starting with a number or space
            elif line[0].isdigit() or line[0] == ' ' or line[0] == '\t':
                continue
            else:
                recall.clear()
                time.clear()
                
                
    return recall, time
    


source = './results/'
dataset = 'deep'
shuf = "_1th_new_metric_level"
# shuf = '_1th'
ef = 500
# M = datsets_map[dataset][0]
M = 16
KNSW = 16
path = os.path.join(source, dataset)
hnsw_avg_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain{shuf}')
hnsw_easy_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain{shuf}_easy')
hnsw_hard_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain{shuf}_hard')
pnsw_path = os.path.join(path, f'{dataset}_ef{ef}_K{KNSW}.pnsw.log')
pnsw_hard_path = os.path.join(path, f'{dataset}_ef{ef}_K{KNSW}.pnsw.log_hard')
pnsw_easy_path = os.path.join(path, f'{dataset}_ef{ef}_K{KNSW}.pnsw.log_easy')

hnsw_k_occur_harder_than_lid_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_1th_k_occur_harder_than_lid')
hnsw_lid_harder_than_k_occur_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_1th_lid_harder_than_k_occur')
hnsw_compact_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_1th_compact')
hnsw_compact_low_k_occur_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_1th_compact_low_k_occur')
hnsw_compact_high_k_occur_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_1th_compact_high_k_occur')
hnsw_high_k_occur_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_1th_high_k_occur')
hnsw_contrastive_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_1th_contrastive')
hnsw_compact_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_1th_compact')
hnsw_lid_hard_3k_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_1th_lid_hard_3k')
hnsw_me_delta0_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_me_delta0_level0')
hnsw_lid_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_lid_level0')
# hnsw_avg_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_1th_lid_level4')

k = 20
width = 2
# hnsw_easy = resolve_log(hnsw_easy_path)
# hnsw_hard = resolve_log(hnsw_hard_path)
# hnsw_avg = resolve_log(hnsw_avg_path)
hnsws =[]
hnsw_lid = resolve_log(hnsw_lid_path)
hnsw_me_delta0 = resolve_log(hnsw_me_delta0_path)
# selected_hnsw = resolve_log(hnsw_lid_hard_3k_path)
# hnsws.append(resolve_log(hnsw_lid_hard_3k_path))
# hnsws.append(resolve_log(hnsw_lid_harder_than_k_occur_path))
# hnsws.append(resolve_log(hnsw_contrastive_path))
# hnsws.append(resolve_log(hnsw_compact_path))
# hnsws.append(resolve_log(hnsw_high_k_occur_path))
# hnsws.append(resolve_log(hnsw_compact_path))
# hnsws.append(resolve_log(hnsw_compact_low_k_occur_path))
# hnsws.append(resolve_log(hnsw_compact_high_k_occur_path))
# selected_hnsw = resolve_log(hnsw_k_occur_harder_than_lid_path)
# selected_hnsw = resolve_log(hnsw_lid_harder_than_k_occur_path)
# avg_hnsw = resolve_log(hnsw_avg_path)
# hnsws.append(resolve_log(hnsw_avg_path))
# hnsws.append(resolve_log(hnsw_k_occur_harder_than_lid_path))
# for i in range(10):
#     hnsws.append(resolve_log(hnsw_avg_path + str(i)))
# pnsw_easy = resolve_python_log(pnsw_easy_path)
# pnsw_hard = resolve_python_log(pnsw_hard_path)
# pnsw_avg = resolve_python_log(pnsw_path)
# plt.plot(hnsw[1], hnsw[0], **plot_info[0], label='HNSW', markersize=20)
# plt.plot(plain[1], plain[0], **plot_info[1], label='HNSW$_0$', markersize=20, color='steelblue')

plt.plot(hnsw_lid[1], hnsw_lid[0], **plot_info[0], label='lid')
plt.plot(hnsw_me_delta0[1], hnsw_me_delta0[0], **plot_info[1], label=r'$\hat{ME^{\forall}_{\delta_0}}$@0.98')


# plt.plot(hnsws[0][1], hnsws[0][0], marker='o', markersize=8, linewidth=width, label='small range', color='salmon' )
# plt.plot(hnsws[1][1], hnsws[1][0], marker='*', markersize=9, linewidth=width, label='large range', color='darkgreen')
# plt.plot(hnsws[2][1], hnsws[2][0], marker='D', markersize=8, linewidth=width, label='low k-occur', color='steelblue')
# plt.plot(hnsws[3][1], hnsws[3][0], marker='^', markersize=8, linewidth=width, label='high k-occur', color='darkgray')

# plt.plot(selected_hnsw[1], selected_hnsw[0],marker='D', markersize=8,linewidth=width, label='k_occur_harder_than_lid', color='darkgreen')
# plt.plot(selected_hnsw[1], selected_hnsw[0],marker='D', markersize=8,linewidth=width, label='lid harder than k-occur', color='darkgreen')
# plt.plot(avg_hnsw[1], avg_hnsw[0], marker='o', markersize=8, linewidth=width, label='Parent set', color='salmon' )


# plt.plot(hnsws[0][1], hnsws[0][0], marker='o', markersize=8, linewidth=width, label='Level 10', color='salmon' )
# plt.plot(hnsws[1][1], hnsws[1][0],marker='+', markersize=16,linewidth=width, label='Level 9', color='salmon')
# plt.plot(hnsws[2][1], hnsws[2][0],marker='*', markersize=16,linewidth=width, label='Level 8', color='salmon')
# plt.plot(hnsws[3][1], hnsws[3][0],marker='o', markersize=8,linewidth=width, label='Level 7', color='darkgray')
# plt.plot(hnsws[4][1], hnsws[4][0],marker='+', markersize=8,linewidth=width, label='Level 6', color='darkgray')
# plt.plot(hnsws[5][1], hnsws[5][0],marker='*', markersize=8,linewidth=width, label='Level 5', color='darkgray')
# plt.plot(hnsws[6][1], hnsws[6][0],marker='o', markersize=5,linewidth=width, label='Level 4', color='steelblue')
# plt.plot(hnsws[7][1], hnsws[7][0],marker='+', markersize=8,linewidth=width, label='Level 3', color='steelblue')
# plt.plot(hnsws[8][1], hnsws[8][0],marker='*', markersize=8,linewidth=width, label='Level 2', color='steelblue')
# plt.plot(hnsws[9][1], hnsws[9][0],marker='o', markersize=8,linewidth=width, label='Level 1', color='darkgreen')

# plt.plot(hnsws[4][1][:-4], hnsws[4][0][:-4],marker='+', markersize=8,linewidth=width, label='Level 6', color='darkgray')
# plt.plot(hnsws[5][1][:-4], hnsws[5][0][:-4],marker='*', markersize=8,linewidth=width, label='Level 5', color='darkgray')
# plt.plot(hnsws[6][1][:-4], hnsws[6][0][:-4],marker='o', markersize=5,linewidth=width, label='Level 4', color='steelblue')
# plt.plot(hnsws[7][1][:-4], hnsws[7][0][:-4],marker='+', markersize=8,linewidth=width, label='Level 3', color='steelblue')
# plt.plot(hnsws[8][1][:-4], hnsws[8][0][:-4],marker='*', markersize=8,linewidth=width, label='Level 2', color='steelblue')
# plt.plot(hnsws[9][1][:-4], hnsws[9][0][:-4],marker='o', markersize=8,linewidth=width, label='Level 1', color='darkgreen')

# plt.plot(hnsw_easy[1], hnsw_easy[0], marker='o', markersize=8, linewidth=width, label='(HNSW) 200 Easy-queries', color='salmon' )
# plt.plot(hnsw_avg[1], hnsw_avg[0],marker='+', markersize=16,linewidth=width, label='(HNSW) 2000 origin-queries', color='salmon')
# plt.plot(hnsw_hard[1], hnsw_hard[0],marker='*', markersize=16,linewidth=width, label='(HNSW) 200 Hard-queries', color='salmon')
# plt.plot(pnsw_easy[1], pnsw_easy[0],marker='o', markersize=8,linewidth=width, label='(pNSW) 200 Easy-queries',color='darkgray')
# plt.plot(pnsw_avg[1], pnsw_avg[0],marker='+', markersize=16,linewidth=width, label='(pNSW) 2000 origin-queries',color='darkgray')
# plt.plot(pnsw_hard[1], pnsw_hard[0],marker='*', markersize=16,linewidth=width, label='(pNSW) 200 Hard-queries',color='darkgray')

# plt.legend(loc="best", fontsize=15)
plt.xticks()
plt.yticks()
plt.xlabel("NDC")
plt.ylabel(r"Recall@50")

plt.legend(loc='best')

# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3,mode="expand", borderaxespad=0.)  #显示图中左上角的标识区域
# plt.title('rand-256-100m (100GB)', fontsize=20)
# plt.legend(loc='best',ncol = 2, fontsize=16)  #显示图中左上角的标识区域
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=6,mode="expand", borderaxespad=0.,fontsize=15.5)  #显示图中左上角的标识区域

plt.tight_layout()
# plt.savefig(f'./figures/bulletin-{dataset}-dist2mean.png',  format='png')
# plt.savefig(f'./figures/{dataset}-outlink-density.png',  format='png')
# check the path exists and if not create it
if not os.path.exists(f'./figures/{dataset}'):
    os.makedirs(f'./figures/{dataset}')
print(f'save to ./figures/{dataset}/{dataset}{shuf}-easy-hard-queries.png')
plt.savefig(f'./figures/{dataset}/{dataset}{shuf}-easy-hard-queries.png',  format='png')
# print(f'save to ./figures/{dataset}/{dataset}-easy-hard-queries-methods.png')
# plt.savefig(f'./figures/{dataset}/{dataset}-easy-hard-queries-methods.png',  format='png')
