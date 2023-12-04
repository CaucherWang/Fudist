import os
from utils import *



def generate_matrix(lowdim, D):
    # generate random matrix from a unit sphere (-1,1) subject to uniform distribution
    return np.random.uniform(-1, 1, size=(D, lowdim))


source = './data/'
dataset = 'rand100'
query_num = 10000
if __name__ == "__main__":
    
    cardinality = 100000000
    # for dim in [50, 100, 150, 200, 250, 300, 500, 750, 1000, 2000, 4000]:
    for dim in [100]:
        np.random.seed(int(time.time()))
        # X = generate_matrix(dim, cardinality)
        Q = generate_matrix(dim, query_num)
        
        # path
        path = os.path.join(source, dataset)
        # create dir if not exists
        if not os.path.exists(path):
            os.makedirs(path)
        # data_path = os.path.join(path, f'{dataset}{dim}_base.fbin')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        
        # to_fvecs(data_path, X)
        
        # write_fbin(data_path, X)
        write_fvecs(query_path, Q)



        
        


