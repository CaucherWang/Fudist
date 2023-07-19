from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os

docs = load_dataset(f"Cohere/miracl-en-corpus-22-12", split="train", streaming=True)

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
    
i = 0
num = 1002000
data = []
for doc in docs:
    if i >= num:
        break
    emb = doc['emb']
    data.append(emb)
    i+=1
    if(i % 50000 == 0):
        print(i)
data_np = np.array(data, dtype=np.float32)
to_fvecs2('/home/hadoop/wzy/dataset/miracl-22-12-cohere1m/miracl-22-12-cohere1m_base.fvecs', data[:1000000])
to_fvecs2('/home/hadoop/wzy/dataset/miracl-22-12-cohere1m/miracl-22-12-cohere1m_query.fvecs', data[1000000:])