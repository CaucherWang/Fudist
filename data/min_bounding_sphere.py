import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import os

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

def min_bounding_sphere(X):
    """
    使用 Welzl 算法求解高维数据集 X 的最小外接球。
    :param X: 高维数据集，每行表示一个数据点。
    :return: 最小外接球的圆心和半径。
    """
    def _bounding_sphere(X, R=[], B=[]):
        if len(X) == 0 or len(R) == K:
            if len(R) == K:
                C, R = _sphere_from_points(B)
                if np.all(np.linalg.norm(X - C, axis=1) <= R):
                    return C, R
            return None, np.inf
        p = X[0]
        X = X[1:]
        C, r = _bounding_sphere(X, R, B)
        if np.linalg.norm(p - C) > r:
            B.append(p)
            C, r = _bounding_sphere(X, R + [p], B)
            B.remove(p)
        return C, r

    def _sphere_from_points(X):
        if len(X) == 1:
            return X[0], 0
        if len(X) == 2:
            return (X[0] + X[1]) / 2, np.linalg.norm(X[0] - X[1]) / 2
        D = X.shape[1]
        C = np.mean(X, axis=0)
        Q = np.zeros((D+1, D+1))
        Q[:D,:D] = 2 * np.dot(X.T, X)
        Q[:D,-1] = -2 * np.sum(X, axis=0)
        Q[-1,:D] = -2 * np.sum(X, axis=0)
        Q[-1,-1] = len(X)
        A = np.zeros((D+1, D+1))
        A[:D,:D] = np.eye(D) * 2
        A[-1,-1] = 0
        b = np.zeros(D+1)
        b[-1] = 1
        y = np.linalg.solve(Q + A, b)
        return C, np.sqrt(np.sum(y[:-1] ** 2) - y[-1])

    # 将数据集 X 转换为 numpy 数组。
    X = np.asarray(X)
    # 获取数据集 X 的维度。
    K = X.shape[1]
    # 使用 Delaunay 三角剖分获取数据集 X 的凸包。
    hull = Delaunay(X)
    simplices = hull.simplices
    print("get Delaunay Triangles")
    # 获取凸包的顶点。
    vertices = np.unique(simplices.ravel())
    print("get ConvexHull Vertices")
    # 获取凸包的边界点。
    boundary_points = np.setdiff1d(vertices, hull.neighbors[vertices])
    print("get ConvexHull Boundary Points")
    # 使用 ConvexHull 获取凸包的表面。
    hull = ConvexHull(X[boundary_points])
    print("get ConvexHull Surface")
    # 获取凸包表面的顶点。
    surface_points = boundary_points[hull.vertices]
    print("get ConvexHull Surface Points")
    # 使用 Welzl 算法求解最小外接球。
    center, radius = _bounding_sphere(X[surface_points])
    return center, radius

source = './data/'
datasets = ['glove1.2m']
dataset = datasets[0]
path = os.path.join(source, dataset)
data_path = os.path.join(path, f'{dataset}_base.fvecs')
X = read_fvecs(data_path)
center, radius = min_bounding_sphere(X)
print(center, radius)