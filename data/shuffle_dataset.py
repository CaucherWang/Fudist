import numpy as np
import os



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
        