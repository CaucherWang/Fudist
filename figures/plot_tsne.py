# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv



def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

import os
source = './data/'
dataset = 'mnist'

def main():
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    # sample 100000 points from data
    query_path = os.path.join(path, f'{dataset}_query.fvecs')

    # data, label, n_samples, n_features = get_data()
    data = read_fvecs(data_path)
    # sample 100000 points from data
    if data.shape[0] > 1000:
        data = data[np.random.choice(data.shape[0], 1000, replace=False), :]
    
    label = [1] * data.shape[0]
    query = read_fvecs(query_path)
    if query.shape[0] > 1000:
        query = query[np.random.choice(query.shape[0], 1000, replace=False), :]

    label2 = [0] * query.shape[0]
    label = label + label2
    data = np.concatenate((data, query), axis=0)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    # label = [0] * len(result)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.savefig(f'./figures/fig/tsne-{dataset}.png',  format='png')


if __name__ == '__main__':
    main()
