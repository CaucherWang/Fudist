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

def skip_diag_strided(A): 
    m = A.shape[0] 
    strided = np.lib.stride_tricks.as_strided 
    s0,s1 = A.strides 
    return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)

source = '/home/hadoop/wzy/dataset/'
datasets = ['yandex-text2image']
if __name__ == '__main__':
    for dataset in datasets:
        dir = os.path.join(source, dataset)
        data_file = os.path.join(dir, f'base.1M.fbin')
        id_gt_file = os.path.join(dir, f'groundtruth.base.1M.base.1k.ibin')
        ood_gt_file = os.path.join(dir, f'groundtruth.base.1M.query.1k.ibin')
        X = fbin_read(data_file)
        id_gt = ibin_read(id_gt_file)[:, :100]
        ood_gt = ibin_read(ood_gt_file)[:, :100]
                
        X_id_gt = X[id_gt]
        X_ood_gt = X[ood_gt]
        
        nq = X_id_gt.shape[0]
        
        id_dists = None
        ood_dists = None
        for i in range(nq):
            tmp1 = X_id_gt[i] @ X_id_gt[i].T
            tmp1 = skip_diag_strided(tmp1)
            tmp11 = tmp1.flatten()
            id_dists = tmp11 if id_dists is None else np.concatenate((id_dists, tmp11))
            tmp2 = X_ood_gt[i] @ X_ood_gt[i].T
            tmp2 = skip_diag_strided(tmp2)
            tmp22 = tmp2.flatten()
            ood_dists = tmp22 if ood_dists is None else np.concatenate((ood_dists, tmp22))
                        
        
        
        # plot the distribution of X_dis, Q_dis, XQ_dist
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot data1
        axes[0].hist(id_dists, bins=30, edgecolor='black')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('id-distrib')
        axes[0].set_ylim(0, 1000000)
        axes[0].set_xlim(0, 1)

        # Plot data2
        axes[1].hist(ood_dists, bins=30, edgecolor='black')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('ood-distrib')
        axes[1].set_ylim(0, 1000000)
        axes[1].set_xlim(0, 1)
        
        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plot
        plt.savefig(f'./figures/{dataset}_inter-gt_distrib-5.png')