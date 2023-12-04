import numpy as np
from hnne import HNNE
from utils import *
import os
import matplotlib.pyplot as plt

# data = np.random.random(size=(1000, 256))

source = './data/'
# datasets = ['deep', 'gist', 'glove1.2m', 'msong', 'sift', 'tiny5m', 'ukbench', 'word2vec']
datasets = ['yandex-text2image1M']

                
if __name__ == "__main__":
    
    for dataset in datasets:
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        data_path_bin = os.path.join(path, f'{dataset}_base.fbin')
        # learn_path = os.path.join(path, f'{dataset}_learn.fbin')
        # query_path = os.path.join(path, f'{dataset}_query.fbin')
        visual_path = os.path.join(path, f'{dataset}_visual.fbin')

        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fbin(visual_path)
        # Q = read_fbin(query_path)
        # T = read_fbin(learn_path)
        # V = np.memmap(visual_path, dtype='float32', mode='r')
        # X = read_fvecs(data_path)
        # X = fbin_read_cnt(data_path, 10000000)
        # X = i8bin_read_cnt(data_path, 10000000)
        D = X.shape[1]
        print(f"{dataset} of dimensionality {D} of cardinality {X.shape[0]}.")
        hnne = HNNE(dim=2)
        projection = hnne.fit_transform(X, verbose=True)
        # write_fbin(os.path.join(path, f'{dataset}_hnne.fbin'), projection)
        # # projection = read_fbin(os.path.join(path, f'{dataset}_hnne.fbin'))
        labels = ['red' if i < 1000000 else 'blue' for i in range(projection.shape[0])]
        # plt.figure(figsize=(10, 10))
        # plt.scatter(*projection.T, s=1, cmap='Spectral')
        # plt.legend(loc= 'best')
        # plt.savefig(f'./data/{dataset}_hnne.png')
        
        
        partitions = hnne.hierarchy_parameters.partitions
        partition_sizes = hnne.hierarchy_parameters.partition_sizes
        print(partition_sizes)
        number_of_levels = partitions.shape[1]
        print(number_of_levels)

        _, ax = plt.subplots(1, 5, figsize=(10*(4 + 1), 10))

        ax[0].set_title('Unlabelled data')
        ax[0].scatter(*projection.T, s=1)

        for i in range(1, 4):
            partition_idx = number_of_levels - i
            ax[i].set_title(f'Partition of level {i}: {partition_sizes[partition_idx]} clusters')
            ax[i].scatter(*projection.T, s=1, c=partitions[:, partition_idx], cmap='Spectral')
        ax[4].scatter(*projection.T, s=1, c=labels, cmap='Spectral')
        plt.savefig(f'./data/{dataset}_hnne_partitions.png')