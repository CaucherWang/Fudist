# encoding=utf-8
import os
import numpy as np
from pprint import pprint
from utils import *

fig = plt.figure(figsize=(8,4))

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
    
def resolve_lb_recall_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        ndcs = []
        recalls = []
        for line in lines:
            if line.startswith('Targets:'):
                ndcs = []
                recalls = []
                line = line[8:].strip()
                line = line.split(',')[:-1]
                recalls = [int(x) for x in line]
            elif len(line)< 100: 
                continue
            else:
                line = line.strip()
                line = line.split(',')[:-1]
                line = [int(x) for x in line]
                ndcs.append(line)
        return ndcs, recalls


source = './results/'
dataset = 'deep'
shuf = "_1th_new_metric_level"
# shuf = '_1th'
ef = 500
# M = datsets_map[dataset][0]
M = 16
KNSW = 16
path = os.path.join(source, dataset)
hnsw_me_delta0_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_lb_recall.log_plain_me_delta0_level9')
hnsw_lid_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_lb_recall.log_plain_lid_level9')
# hnsw_avg_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_1th_lid_level4')

k = 50
width = 2
# hnsw_easy = resolve_log(hnsw_easy_path)
# hnsw_hard = resolve_log(hnsw_hard_path)
# hnsw_avg = resolve_log(hnsw_avg_path)
hnsws =[]
hnsw_lid, recalls = resolve_lb_recall_log(hnsw_lid_path)
hnsw_me_delta0, recalls = resolve_lb_recall_log(hnsw_me_delta0_path)

ndc_lid = hnsw_lid[-1]
ndc_me_delta0 = hnsw_me_delta0[-1]

# Define the number of bins
num_bins = 50

# Define the range of the histogram
start = 0
end = 50000

# Calculate the width of each bin
bin_width = (end - start) / num_bins

# Define the bin edges
bin_edges = np.arange(start, end + bin_width, bin_width)

# Plot the histogram
plt.hist(ndc_lid, bins=bin_edges, edgecolor='black')

plt.xlim(0, 50000)
plt.ylim(0, 220)
plt.xlabel('NDC when LB recall@50=0.98')
plt.ylabel('#queries')
plt.tight_layout()
plt.savefig(f'./figures/{dataset}/{dataset}{shuf}-lid-ndc-hist.png',  format='png')
print(f'save to ./figures/{dataset}/{dataset}{shuf}-lid-ndc-hist.png')
# data = {
#     'lid':{},
#     'me_delta0':{},
# }
# for i in range(len(recalls)):
#     data['me_delta0'][recalls[i]] = list(hnsw_me_delta0[i])
#     data['lid'][recalls[i]] = list(hnsw_lid[i])

# # Sort the dictionary by keys (recall values)
# for key in data:
#     data[key] = dict(sorted(data[key].items()))
# xs = np.array(recalls) / k * 100

    

# # plot the violin graph
# parts = plt.violinplot(list(data['me_delta0'].values()),positions=xs , showmeans=False, showmedians=True, showextrema=False, widths=1.5)
# # for pc in parts['bodies']:
# #     pc.set_facecolor('salmon')
# mean_ndc = np.mean(list(data['me_delta0'].values()), axis=1)
# plt.plot(xs, mean_ndc, **plot_info[5], label=r'$\hat{ME^{\forall}_{\delta_0}}$@0.98')

# parts = plt.violinplot(list(data['lid'].values()),positions=xs , showmeans=False, showmedians=True, showextrema=False, widths=1.5)
# # for pc in parts['bodies']:
# #     pc.set_facecolor('blue')
# mean_ndc = np.mean(list(data['lid'].values()), axis=1)
# plt.plot(xs, mean_ndc, **plot_info[1], label='lid')


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
# plt.xticks()
# plt.yticks()
# plt.ylim(0, 20000)
# plt.xlabel(f"LB Recall@{k}")
# plt.ylabel(r"NDC")

# plt.legend(loc='best')

# # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3,mode="expand", borderaxespad=0.)  #显示图中左上角的标识区域
# # plt.title('rand-256-100m (100GB)', fontsize=20)
# # plt.legend(loc='best',ncol = 2, fontsize=16)  #显示图中左上角的标识区域
# # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=6,mode="expand", borderaxespad=0.,fontsize=15.5)  #显示图中左上角的标识区域

# plt.tight_layout()
# # plt.savefig(f'./figures/bulletin-{dataset}-dist2mean.png',  format='png')
# # plt.savefig(f'./figures/{dataset}-outlink-density.png',  format='png')
# # check the path exists and if not create it
# if not os.path.exists(f'./figures/{dataset}'):
#     os.makedirs(f'./figures/{dataset}')
# print(f'save to ./figures/{dataset}/{dataset}{shuf}-easy-hard-violin.png')
# plt.savefig(f'./figures/{dataset}/{dataset}{shuf}-easy-hard-violin.png',  format='png')
# # print(f'save to ./figures/{dataset}/{dataset}-easy-hard-queries-methods.png')
# # plt.savefig(f'./figures/{dataset}/{dataset}-easy-hard-queries-methods.png',  format='png')
