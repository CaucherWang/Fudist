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

def get_new_pos(id, landmarks):
    pos = np.searchsorted(landmarks, id, side='left')
    return id + pos

def find_row_within_error(matrix, factor, error):
    diff = np.abs(matrix - factor)  # Calculate the absolute difference between each element and the factor
    row_errors = np.max(diff, axis=1)  # Find the maximum difference in each row
    matching_rows = np.where(row_errors <= error)[0]  # Find the rows with differences within the error threshold

    return matching_rows


path = os.path.join(source, dataset)
data_path = os.path.join(path, f'{dataset}_base.fbin')
train_path = os.path.join(path, f'{dataset}_learn.fbin')
gt_path = os.path.join(path, f'{dataset}_groundtruth.ibin')
sample_percent = 0.01
result_path = os.path.join(path, f'{dataset}_base.fbin-ood{sample_percent}')
pos_path = os.path.join(path, f'train_pos_ood{sample_percent}.ibin')
new_gt_path = os.path.join(path, f'{dataset}_groundtruth.ibin-ood{sample_percent}')


# X = read_fbin(data_path)
# X_size = X.shape[0]
# new_X = read_fbin(result_path)
GT = read_ibin(gt_path)
nq, nk = GT.shape
train_positions = read_ibin_simple(pos_path)

landmarks = [train_positions[i] - i - 1 for i in range(len(train_positions))]

for i in range(nq):
    for j in range(nk):
        GT[i][j] = get_new_pos(GT[i][j], landmarks)

write_ibin(new_gt_path, GT)