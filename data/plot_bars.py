from utils import *

kgraph = np.array([10 ** 6, 10**6, 1193514, 10**6, 10**6, 10**6]) * 32
nsw = np.array([31999728,31999728,38192176,31999728,31999728,31999728,])
pnsw = np.array([24182559,24752398,22584566,19936728,21865762,19898903,])
hnsw = np.array([22613026,22206985,6412385,3509681,20667089,17923231,])
nsg = np.array([27279045,25756063,3432993,3258979,28921386,16743676])

nsw = nsw / kgraph
pnsw = pnsw / kgraph
hnsw = hnsw / kgraph
nsg = nsg / kgraph
kgraph = kgraph / kgraph

x = ['Deep', 'SIFT', 'Glove', 'Word2Vec', 'Rand200', 'Gauss200']

# plot bars in one bar
# positions = [1,1.15,1.3,1.45,1.6, 2,2.15,2.3,2.45,2.6, 3,3.15,3.3,3.45,3.6, 4,4.15,4.3,4.45,4.6]
positions = [1,2,3,4, 5, 6]
width = 0.15

plt.figure(figsize=(10, 5))
plt.bar([pos-0.3 for pos in positions],  kgraph,width,label='kGraph', color='dimgray',edgecolor='black', hatch='/')
plt.bar([pos-0.15 for pos in positions],  nsw,width,label='NSW', color='lavender',edgecolor='black', hatch='*')

plt.bar([pos for pos in positions],  pnsw,width,label='pNSW', color='gray',edgecolor='black', hatch='+')
plt.bar([pos+0.15 for pos in positions],  hnsw,width,label='HNSW', color='whitesmoke',edgecolor='black', hatch='\\')

plt.bar([pos+0.3 for pos in positions],  nsg,width,label='NSG', color='silver',edgecolor='black', hatch='.')



# plt.xlabel('', fontsize=15)
plt.xticks(positions, x,fontsize=16 )
# plt.xticks(positions, ['DSTree', 'iSAX2+' ,'TARDIS','Dumpy','Dumpy-f', 'DSTree', 'iSAX2+','TARDIS',  'Dumpy','Dumpy-f','DSTree', 'iSAX2+','TARDIS',  'Dumpy','Dumpy-f','DSTree', 'iSAX2+','TARDIS',  'Dumpy','Dumpy-f',],fontsize=12, rotation=90)

plt.yticks(fontsize=16)
plt.ylabel('#edges',fontsize=16)
# plt.legend(loc='upper left',  ncol=3 ,borderaxespad=0.,fontsize=12, framealpha=0.2)  #显示图中左上角的标识区域

# ax.set_title('this is title')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=5,mode="expand", borderaxespad=0.,fontsize=15)  #显示图中左上角的标识区域

plt.tight_layout()
plt.savefig(f'./figures/edgesnum.png',  format='png')
print('save fig to ./figures/edgesnum.png')