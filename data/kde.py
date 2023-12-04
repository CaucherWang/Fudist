import numpy as np
from sklearn.neighbors import KernelDensity
import os
from utils import *
from joblib import dump, load
from pprint import pprint
from sklearn.model_selection import GridSearchCV, LeaveOneOut

# Generate a high-dimensional dataset
source = '../data/'
datasets = ['deep']

dataset = datasets[0]
path = os.path.join(source, dataset)
data_path = os.path.join(path, f'{dataset}_base.fvecs')
X  =read_fvecs(data_path)
print(X.shape)

# bandwidth = 0.2
# # Create a KDE object with a Gaussian kernel
# kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
# print('start fitting')
# # Fit the KDE to the dataset
# kde.fit(X)
# print('fitting done')
bws = np.linspace(0.1, 10, 100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),{'bandwidth': bws}, cv=LeaveOneOut())
grid.fit(X)
pprint(grid.best_params_, grid.best_score_, grid.cv_results_['mean_test_score'])
pprint(grid)
# dump(kde, f'{dataset}_kde_{bandwidth}.joblib')

# kde = load(f'{dataset}_kde_{bandwidth}.joblib')

# Evaluate the KDE at some points
x_eval = X[:100]
pprint.pprint(x_eval)
log_density = kde.score_samples(x_eval)

print(log_density)
