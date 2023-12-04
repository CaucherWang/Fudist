from utils import *
import os
import random
import numpy as np


# Load the data
source = '/home/hadoop/wzy/dataset/'
# datasets = ['deep', 'gist', 'glove1.2m', 'msong', 'sift', 'tiny5m', 'ukbench', 'word2vec']
dataset = 'yandex-text2image1M'

def get_random_positions(num, size):
    return random.sample(range(size), num)


def insert_rows(A, B, positions):
    # Check if the number of positions matches the number of rows in B
    if len(positions) != B.shape[0]:
        raise ValueError("Number of positions should match the number of rows in B.")

    # Create an empty result matrix with the appropriate size
    result = np.empty((A.shape[0] + B.shape[0], A.shape[1]), dtype=A.dtype)
    
    curA = 0
    curB = 0
    for i in range(result.shape[0]):
        if curB < B.shape[0] and i == positions[curB]:
            result[i] = B[curB]
            curB += 1
        else:
            result[i] = A[curA]
            curA += 1
    assert curA == A.shape[0]
    assert curB == B.shape[0]

    return result

def count_elements_not_bigger_than(sorted_list, target):
    left, right = 0, len(sorted_list) - 1
    count = -1  # Initialize count to -1

    while left <= right:
        mid = left + (right - left) // 2

        if sorted_list[mid] <= target:
            count = mid + 1  # Count all elements up to and including mid
            left = mid + 1
        else:
            right = mid - 1

    return count

def get_new_pos(id, train_positions):
    if id < train_positions[0]:
        return id
    if id >= train_positions[-1]:
        return id + len(train_positions)
    return id + count_elements_not_bigger_than(train_positions, id)

path = os.path.join(source, dataset)
data_path = os.path.join(path, f'{dataset}_base.fbin')
train_path = os.path.join(path, f'{dataset}_learn.fbin')
gt_path = os.path.join(path, f'{dataset}_groundtruth.ibin')
sample_percent = 0.01
result_path = os.path.join(path, f'{dataset}_base.fbin-ood{sample_percent}')


X = read_fbin(data_path)
X_size = X.shape[0]
sample_size = int(X_size * sample_percent)
T = read_fbin_cnt(train_path, sample_size)
GT = read_ibin(gt_path)
nq, nk = GT.shape

print(f"read finished. X shape: {X.shape}")

train_positions = np.array(get_random_positions(sample_size, X_size + sample_size))
train_positions.sort()

result = insert_rows(X, T, train_positions)

print(f"insert finished. result shape: {result.shape}")

pos_path = os.path.join(path, f'train_pos_ood{sample_percent}.ibin')
write_fbin(result_path, result)
write_ibin_simple(pos_path, train_positions)

print('write finished')

# iterate all values in GT
for i in range(nq):
    for j in range(nk):
        old_id = GT[i][j]
        GT[i][j] = get_new_pos(GT[i][j], train_positions)
        assert(np.array_equal(result[GT[i][j]], X[old_id]))
        
print('GT update finished')
        
new_gt_path = os.path.join(path, f'{dataset}_groundtruth.ibin-ood{sample_percent}')
write_ibin(new_gt_path, GT)