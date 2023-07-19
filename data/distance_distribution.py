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


# source = '/home/hadoop/wzy/dataset/'
# datasets = ['yandex-text2image1M']
source  = './data/'
datasets = ['sift']
if __name__ == '__main__':
    for dataset in datasets:
        dir = os.path.join(source, dataset)
        data_file = os.path.join(dir, f'{dataset}_base.fbin')
        train_file = os.path.join(dir, f'{dataset}_learn.fbin')
        query_file = os.path.join(dir, f'{dataset}_query.fbin')
        X = fbin_read(data_file)
        Q = fbin_read(query_file)
        T = fbin_read(train_file)
        X_sampl1 = rand_select(1000, X.shape[0])
        X_sampl2 = rand_select(1000, X.shape[0])
        Q_sampl1 = rand_select(1000, Q.shape[0])
        Q_sampl2 = rand_select(1000, Q.shape[0])
        T_sampl1 = rand_select(1000, T.shape[0])
        T_sampl2 = rand_select(1000, T.shape[0])
        
        X1 = X[X_sampl1]
        X2 = X[X_sampl2]
        Q1 = Q[Q_sampl1]
        Q2 = Q[Q_sampl2]
        T1 = T[T_sampl1]
        T2 = T[T_sampl2]
        
        X_dis = X1 @ X2.T
        Q_dis = Q1 @ Q2.T
        T_dis = T1 @ T2.T
        XQ_dist = X1 @ Q1.T
        
        
        
        # flatten X_dis, Q_dis, XQ_dist
        X_dis = X_dis.flatten()
        Q_dis = Q_dis.flatten()
        T_dis = T_dis.flatten()
        XQ_dist = XQ_dist.flatten()
        
        # plot the distribution of X_dis, Q_dis, XQ_dist
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Plot data1
        axes[0][0].hist(X_dis, bins=30, edgecolor='black')
        axes[0][0].set_xlabel('Value')
        axes[0][0].set_ylabel('Frequency')
        axes[0][0].set_title('Base-Base')

        # Plot data2
        axes[0][1].hist(Q_dis, bins=30, edgecolor='black')
        axes[0][1].set_xlabel('Value')
        axes[0][1].set_ylabel('Frequency')
        axes[0][1].set_title('Query-Query')
        
                # Plot data3
        axes[1][0].hist(T_dis, bins=30, edgecolor='black')
        axes[1][0].set_xlabel('Value')
        axes[1][0].set_ylabel('Frequency')
        axes[1][0].set_title('Train-Train')


        # Plot data3
        axes[1][1].hist(XQ_dist, bins=30, edgecolor='black')
        axes[1][1].set_xlabel('Value')
        axes[1][1].set_ylabel('Frequency')
        axes[1][1].set_title('Base-Query')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plot
        plt.savefig(f'./figures/{dataset}_dist.png')