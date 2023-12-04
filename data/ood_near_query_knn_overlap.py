from utils import *
from queue import PriorityQueue
import os
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,4.8))
plt.rcParams['mathtext.fontset'] = "stix"
plt.rcParams['font.family'] = 'calibri'
# plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 22

    
source = './data/'
dataset = 'yandex-text2image1M'
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs')

    X = read_fvecs(base_path)
    Q = read_fvecs(query_path)
    Q_sample = Q[:500]
    GT = read_ivecs(GT_path)[:,:100]
    
    q_gt, q_gt_dis = compute_GT_CPU_IP(Q, Q_sample, 2) 
    
    
    intersec_num = []
    for i in range(len(Q_sample)):
        q = Q_sample[i]
        q_couple = q_gt[i][1] if q_gt[i][0] == i else q_gt[i][0]
        q_knn = GT[i]
        q_couple_knn = GT[q_couple]
        intersec = np.intersect1d(q_knn, q_couple_knn)
        intersec_num.append(len(intersec))
        
    print(np.mean(intersec_num))
    
    
    
    
