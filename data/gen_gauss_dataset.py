import os
import numpy as np
import time
from utils import *
source = './data/'

def generate_matrix(lowdim, D):
    return np.random.normal(size=(D, lowdim))


dataset = 'gauss100'
query_num = 10000
if __name__ == "__main__":
    
    cardinality = 100000000
    for dim in [100]:
        np.random.seed(int(time.time()))
        # X = generate_matrix(dim, cardinality)
        Q = generate_matrix(dim, query_num)
        
        # path
        path = os.path.join(source, dataset)
        # create if not exists
        if not os.path.exists(path):
            os.makedirs(path)
        # data_path = os.path.join(path, f'{dataset}{dim}_base.fbin')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        
        # write_fbin(data_path, X)
        write_fvecs(query_path, Q)



        
        


