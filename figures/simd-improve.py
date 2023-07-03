import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5.5, 3.2))

# gist
# ratios = {
#     'PCA':[63.6, 93.2],
#     'DWT':[20.6, 93.2],
#     'ADS': [38.5, 93.1],
#     'LSH': [22.4, 28.6],
#     'FINGER':[54.6, 61.9],
#     'OPQ': [34.1, 34.9],
    
# }


# imagenet
ratios = {
    'HNSW':[10572.3, 6021.63],
    'PCA':[6657, 5298],
    'DWT':[8900, 5697],
    'ADS': [7763, 5196],
    'LSH': [9587, 6028],
    'FINGER':[16870, 11814],
    'OPQ': [5941, 3806],
    'SEANET':[16333, 9905],
}

res = []
for ratio in ratios.values():
    res.append(ratio[1] / ratio[0])

# fig, ax = plt.subplots()

# plot bars with ratios
plt.bar(ratios.keys(), res, color='darkgray', edgecolor='black')

# plt.bar(labels, read ,width,bottom=write + cpu, label='Read', color='dimgray', edgecolor='black', hatch='/')
# plt.bar(labels, write ,width,bottom=cpu, label='Write', edgecolor='black', color='silver', hatch='.')
# plt.bar(labels, cpu ,width,label='CPU', color='whitesmoke', edgecolor='black', hatch='\\')
# plt.xlabel('Rand100m', fontsize=16)
plt.xticks(fontsize=16)
# plt.yscale('log')
plt.yticks(fontsize=16)
plt.ylabel('Time reduced (%)',fontsize=14)
# ax.set_title('this is title')
# plt.legend(loc='best',fontsize=18)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3,mode="expand", borderaxespad=0.,fontsize=16)  #显示图中左上角的标识区域
plt.tight_layout()
# plt.show()
plt.savefig(f'./figures/fig/simd-improved-gist.png',  format='png')
