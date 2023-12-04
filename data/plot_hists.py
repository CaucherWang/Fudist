import matplotlib.pyplot as plt
import numpy as np

def fbin_read(fname: str):
    dtype = np.dtype('>f4') 
    # return np.memmap(fname, dtype=dtype, mode='r')
    # return np.fromfile(fname , dtype=dtype)
    return np.loadtxt(fname)


dataset = 'glove1.2m'
path = f'./results/{dataset}_hop_distances.fbin'
distances = fbin_read(path)
plt.figure(figsize=(6, 5))
plt.hist(distances, bins=100, edgecolor = 'black')
plt.title(f'Distrib. of length of used edges in {dataset}1M, M=16')
plt.ylabel('Frequency')
plt.xlabel('Distance')
plt.tight_layout()
plt.savefig(f'./figures/hist_{dataset}_used_edges.png')

