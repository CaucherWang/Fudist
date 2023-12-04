from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import LocallyLinearEmbedding
import os
import numpy as np
from utils import *
import os
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool


source = './data/'
dataset = 'rand'
for dim in [4000]:
    print('dim', dim)
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}{dim}_base.fvecs')

    X = read_fvecs(data_path)
    X = np.unique(X, axis = 0)
    print(X.shape)
    X = X[:100000]
    nnn = 100
    gt_i, gt_d = compute_GT_CPU(X, X, nnn)
    print('gt finished')

    import pandas as pd
    from geomle import mle
    print(mle(pd.DataFrame(X), average = True, k2 = 99, dist = gt_d)[0])

# import skdim
# #generate data : np.array (n_points x n_dim). Here a uniformly sampled 5-ball embedded in 10 dimensions

# #estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point):
# lpca = skdim.id.lPCA().fit_pw(X,
#                               precomputed_knn = gt_i,
#                               n_neighbors = nnn,
#                               n_jobs = 32)
                            
# #get estimated intrinsic dimension
# print(np.mean(lpca.dimension_pw_)) 




# # Define the number of nearest neighbors to use
# n_neighbors = 100

# # Initialize the LLE object
# lle = LocallyLinearEmbedding(n_neighbors=n_neighbors+1, n_components=2)

# # Fit the LLE object to the data
# lle.fit(X)

# # Calculate the locally intrinsic dimensionality using the MLE method
# lid = lle.inverse_transform(lle.transform(X)).sum(axis=1).mean()

# print("The locally intrinsic dimensionality of the dataset is:", lid)

# def l2_distance(a, b):
#     """
#     Compute the L2 distance between two vectors a and b.
#     """
#     return np.sum((a - b)**2)

# def ComputeIntrinsicDimensionality(dataset, SampleQty = 1000000):
#     dist = []
#     DistMean = 0
#     for n in range(SampleQty):
#         r1 = random.randint(0, len(dataset))
#         r2 = random.randint(0, len(dataset))      
#         obj1 = dataset[r1]
#         obj2 = dataset[r2]
#         d = l2_distance(obj1, obj2)
#         dist.append(d)
#         DistMean += d
#     DistMean /= float(SampleQty)
#     DistSigma = 0
#     for i in range(SampleQty):
#         DistSigma += (dist[i] - DistMean) * (dist[i] - DistMean)
#     DistSigma /= float(SampleQty)
#     IntrDim = DistMean * DistMean / (2 * DistSigma)
#     return IntrDim

# print(ComputeIntrinsicDimensionality(X))