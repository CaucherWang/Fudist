import os
import numpy as np
import struct

source = './data/'
datasets = ['gist']

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

if __name__ == "__main__":
    
    for dataset in datasets:
        np.random.seed(0)
        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')

        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        D = X.shape[1]
        # X = X[:2000, :]

        # generate random orthogonal matrix, store it and apply it
        print(f"SVD {dataset} of dimensionality {D}.")
        _,s,U = svd(X)
        # smat = np.diag(s)
        # tmp = np.dot(u, np.dot(smat, vh))
        # print(np.allclose(X, tmp, atol=1e-3))
        # U = UT.T
        # print(check_orthogonal_matrix(UT))
        XU = np.dot(X, U)   # TODO: we don't need to transpose U, right?

        projection_path = os.path.join(path, 'SVD.fvecs')
        transformed_path = os.path.join(path, f'SVD_{dataset}_base.fvecs')

        to_fvecs(projection_path, U)
        to_fvecs(transformed_path, XU)
