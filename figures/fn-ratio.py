import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5.5, 3.2))

ratios = {
    'PCA':0,
    'DWT':0,
    'ADS': 0.03,
    'LSH': 0.36,
    'FINGER':9.0,
    'OPQ': 0.10,
    
}

# fig, ax = plt.subplots()

# plot bars with ratios
plt.bar(ratios.keys(), ratios.values(), color='darkgray', edgecolor='black')

# plt.bar(labels, read ,width,bottom=write + cpu, label='Read', color='dimgray', edgecolor='black', hatch='/')
# plt.bar(labels, write ,width,bottom=cpu, label='Write', edgecolor='black', color='silver', hatch='.')
# plt.bar(labels, cpu ,width,label='CPU', color='whitesmoke', edgecolor='black', hatch='\\')
# plt.xlabel('Rand100m', fontsize=16)
plt.xticks(fontsize=12)
plt.yscale('log')
plt.yticks([0.1,1,10], [0.1,1,10], fontsize=20)
plt.ylabel('False negative ratio (%)',fontsize=18)
# ax.set_title('this is title')
# plt.legend(loc='best',fontsize=18)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3,mode="expand", borderaxespad=0.,fontsize=16)  #显示图中左上角的标识区域
plt.tight_layout()
# plt.show()
plt.savefig(f'./figures/fig/fn-ratio.png',  format='png')
