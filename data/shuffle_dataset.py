import numpy as np
import os
import struct

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


source = '/home/hadoop/wzy/dataset/'
datasets = ['deep1M']
if __name__ == '__main__':
    for dataset in datasets:
        dir = os.path.join(source, dataset)
        data_file = os.path.join(dir, f'{dataset}_base.fbin')
        train_file = os.path.join(dir, f'{dataset}_learn.fbin')
        shuf_data_file = os.path.join(dir, f'{dataset}_base_shuf.fbin')
        pos_file = os.path.join(dir, f'{dataset}_pos.ibin')
        X = fbin_read(data_file)
        T = fbin_read(train_file)
        np.random.shuffle(T)
        nx = X.shape[0]
        nq = T.shape[0]
        
        numbers = rand_select(nq, nx + nq)
        numbers = np.unique(numbers)
        assert numbers.shape[0] == nq
        
        # put Q into X, in the order of numbers
        X_new = np.zeros((nq + nx, X.shape[1]))
        idx = 0
        idxx = 0
        for i in range(nq+nx):
            if i %100000 == 0:
                print(i)
            if idx < numbers.shape[0] and i == numbers[idx]:
                X_new[i] = T[idx]
                idx += 1
            else:
                X_new[i] = X[idxx]
                idxx += 1
        
        
        print(X_new.shape)
        fbin_write(X_new, shuf_data_file)
        ibin_write(numbers, pos_file)
        