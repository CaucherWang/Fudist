import os
import numpy as np
import struct
import pandas as pd
from numba import njit
from sklearn.decomposition import PCA

source = './data/'
datasets = ['gist']
datasets_map = {
    'imagenet': (6, 200),
    # 'msong': (6, 1000),
    'word2vec': (6, 1000),
    # 'ukbench': (8, 200),
    'gist': (8, 1000),
    'deep': (8, 1000),
    # 'glove1.2m': (8, 1000),
    # 'sift': (8, 1000),
    # 'tiny5m': (8, 1000),
    # 'uqv':(8,1000),
    # 'glove-100':(4,1000),
    # 'crawl': (6, 1000),
    # 'enron': (8, 1000),
    'mnist': (8, 1000),
    # 'cifar': (8, 200),
    # 'sun':(8, 200),
    # 'notre':(8, 200),
    # 'nuswide':(4, 200),
    'trevi': (8, 200),
    # 'gauss50':(),
    # 'gauss100':(),
    # 'gauss150':(),
    # 'gauss200':(),
    # 'gauss250':(),
    # 'gauss300':(),
    # 'gauss500':(),
    # 'gauss1000':(),
    # 'gauss2000':(),
    # 'gauss4000':(),

}


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

def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q

def svd(D):
    '''
    get the right singular vectors of a D (m *n), where m is very large while n is small
    '''
    m, n = D.shape
    if m > 1000:
        print('m > 1000, use randomized SVD')
        from sklearn.utils.extmath import randomized_svd
        u, s, vh = randomized_svd(D, n_components=n)
    else:
        print('m < 1000, use numpy SVD')
        u, s, vh = np.linalg.svd(D, full_matrices=False)
    return u, s, vh

def check_orthogonal_matrix(A):
    B = np.dot(A, A.T)
    C = np.eye(A.shape[0])
    return np.allclose(B, C)
    
    
def check_same_matrix(A, B):
    return np.allclose(A, B)

def to_floats(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            a = struct.pack('f', y)
            fp.write(a)

@njit
def calc_approx_dist(X, Q, RealDist):
    # Q: query vector
    # X_code: PQ encoded base vector
    # RealDist: real distance between Q and X
    # return: approximated distance between Q and X / RealdDist
    result = []
    for i in range(Q.shape[0]):
        for j in range(X.shape[0]):
            dist = np.sum((Q[i] - X[j]) ** 2)
            if RealDist[i][j] == 0:
                if dist == 0:
                    result.append(1)
            else:        
                result.append(dist / RealDist[i][j])
    return result



    

if __name__ == "__main__":
    
    for dataset in datasets_map.keys():
        np.random.seed(0)
        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')

        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        D = X.shape[1]

        # generate random orthogonal matrix, store it and apply it
        # print(f"SVD {dataset} of dimensionality {D}.")
        # _,s,U = svd(X)
        # smat = np.diag(s)
        # tmp = np.dot(u, np.dot(smat, vh))
        # print(np.allclose(X, tmp, atol=1e-3))
        # U = UT.T
        # print(check_orthogonal_matrix(UT))
        # XU = np.dot(X, U)
        
             
        # print(f"PCA {dataset} of dimensionality {D}.")
        pca = PCA(n_components=D)
        pca.fit(X)
        s = np.array(pca.singular_values_)
        len = np.linalg.norm(s)
        s = s / len
        # normalize the array s
        print(np.linalg.norm(s))
        
        print(s)
        
        # plot the distribution of s
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.5, 3.2))
        plt.plot(s)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('Singular values',fontsize=14)
        plt.tight_layout()
        plt.savefig(f'./figures/fig/{dataset}_singular_values.png',  format='png')
        
        # U = U.T
        # # print(check_orthogonal_matrix(U))
        # # XU = pca.transform(X)
        # XU = np.dot(X, U)


        # projection_path = os.path.join(path, 'PCA.fvecs')
        # transformed_path = os.path.join(path, f'PCA_{dataset}_base.fvecs')
        # to_fvecs(projection_path, U)
        # to_fvecs(transformed_path, XU)
        

        # sampleQuery = datasets_map[dataset][1]
        # sampleBase = 10000
        # query_path = os.path.join(path, f'{dataset}_query.fvecs')
        # dist_path = os.path.join(path, f'Real_Dist_{sampleBase}_{sampleQuery}.fvecs')
        # sample = 0.5
        # U = read_fvecs(projection_path)
        # lowdim = int(U.shape[0] * sample)
        
        # X = read_fvecs(transformed_path)[:sampleBase, :lowdim]
        # Q = read_fvecs(query_path)[:sampleQuery]
        # Q = np.dot(Q, U)[ : , :lowdim]
        # RealDist = read_fvecs(dist_path)
        # result = calc_approx_dist(X, Q, RealDist)
        # result = np.array(result)
        # result = np.sort(result)
        # result_path = os.path.join(path, f'PCA_{sample}_approx_dist.floats')
        # to_floats(result_path, result)
        
        
