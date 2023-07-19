import numpy as np
import os
import struct
import matplotlib.pyplot as plt

def fbin_read(fname: str):
    a = np.memmap(fname, dtype='int32', mode='r')
    # a = np.fromfile(fname + ".fbin", dtype='int32')
    num = a[0]
    d = a[1]
    print(f"{num} * {d}")
    return a[2:].reshape(-1, d)[:num, :].copy().view('float32')

def fbin_write(x, path: str):
    x = x.astype('float32')
    f = open(path, "wb")
    n, d = x.shape
    np.array([n, d], dtype='int32').tofile(f)
    x.tofile(f)
    
def rand_select(nq, nx):
    # randomly select nq numbers from [0,nx)
    return np.random.choice(nx, nq, replace=False)

def ibin_write(x, path: str):
    x = x.astype('int32')
    f = open(path, "wb")
    x.tofile(f)


def ibin_read(fname: str):
    a = np.fromfile(fname , dtype='int32')
    num = a[0]
    d = a[1]
    return a[2:].reshape(-1, d)[:num, :].copy()

source = '/home/hadoop/wzy/dataset/'
datasets = ['yandex-text2image']
if __name__ == '__main__':
    for dataset in datasets:
        dir = os.path.join(source, dataset)
        id_query_file = os.path.join(dir, f'base.skip.1M.1k.fbin')
        data_file = os.path.join(dir, f'base.1M.fbin')
        ood_query_file = os.path.join(dir, f'query.public.1k.fbin')
        id_gt_file = os.path.join(dir, f'groundtruth.base.1M.base.1k.ibin')
        ood_gt_file = os.path.join(dir, f'groundtruth.base.1M.query.1k.ibin')
        X = fbin_read(data_file)
        id_query = fbin_read(id_query_file)
        ood_query = fbin_read(ood_query_file)
        id_gt = ibin_read(id_gt_file)
        ood_gt = ibin_read(ood_gt_file)
        
        id_gt = id_gt[:, 99].flatten()
        ood_gt = ood_gt[:, 99].flatten()
        
        X_id_gt = X[id_gt]
        X_ood_gt = X[ood_gt]
        
        nq = id_query.shape[0]
        
        id_dists = []
        ood_dists = []
        for i in range(nq):
            id_dists.append(id_query[i] @ X_id_gt[i].T)
            ood_dists.append(ood_query[i] @ X_ood_gt[i].T)    
                        
        
        
        # plot the distribution of X_dis, Q_dis, XQ_dist
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot data1
        axes[0].hist(id_dists, bins=30, edgecolor='black')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('id-distrib')

        # Plot data2
        axes[1].hist(ood_dists, bins=30, edgecolor='black')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('ood-distrib')
        
        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plot
        plt.savefig(f'./figures/{dataset}_gt_distrib-100.png')