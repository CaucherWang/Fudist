from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os

def find_file_by_prefix(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            return os.path.join(directory, filename)
    return None

def to_fvecs2(filename, data):
    print(f"Writing File - {filename}")
    dim = (np.int32)(data.shape[1])
    data = data.astype(np.float32)
    # ff = np.memmap(path, dtype='int32', shape=(len(array), dim+1), mode='w+')
    # ff[:, 0] = dim
    # ff[:, 1:] = array.view('int32')
    # del ff
    dim_array = np.array([dim] * len(data), dtype='int32')
    file_out = np.column_stack((dim_array, data.view('int32')))
    file_out.tofile(filename)
# en_mc4 = load_dataset('mc4', 'en', features=['text'])
# en_mc4.save_to_disk('./curpus/')

def num2str(num):
    if num >= 100:
        return str(num)
    if num >= 10:
        return '0' + str(num)
    return '00' + str(num)

data_dir = '/home/hadoop/wzy/dataset/miracl-en-corpus-22-12/data/'
data_path = 'train-00{num}'
data =None

for i in [10]:
    prefix = data_path.format(num = num2str(i))
    path = find_file_by_prefix(data_dir, prefix)
    tab = pq.read_table(path)
    tab = tab['emb'].to_pandas()
    new_data = np.array(tab.tolist())
    if data is not None:
        data = np.concatenate((data, new_data), axis=0)
    else:
        data = new_data
        
print(data.shape)
rd_data = np.unique(data, axis = 0)
print(rd_data.shape)
to_fvecs2('/home/hadoop/wzy/dataset/miracl-22-12-cohere1m/miracl-22-12-cohere1m_query.fvecs', rd_data)