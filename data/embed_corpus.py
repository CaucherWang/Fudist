from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os
import gzip
import json

def find_file_by_prefix(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            return os.path.join(directory, filename)
    return None

def read_gz(filepath):
    data = []
    with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
        for line in f:
            if line:
                example = json.loads(line)
                text = example['text']
                data.append(text)
    return data

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

data_dir = '/home/hadoop/wzy/ADSampling/corpus/c4/en/'
data_path = 'c4-train.00{num}'
data = []

for i in [10]:
    prefix = data_path.format(num = num2str(i))
    path = find_file_by_prefix(data_dir, prefix)
    # tab = pq.read_table(path)
    # tab = tab['text'].to_pandas().tolist()
    tab = read_gz(path)
    data += tab

print(len(data))
# rd_data = np.unique(data, axis = 0)
# print(rd_data.shape)

# import chromadb
# from chromadb.utils import embedding_functions
# chroma_client = chromadb.Client()
# default_ef = embedding_functions.DefaultEmbeddingFunction()
# item = data[0]
# print(item)
# embed = default_ef(item)
# print(embed)


# from towhee import AutoPipes, AutoConfig
# config = AutoConfig.load_config('sentence_embedding')
# config.model = 'sentence-t5-xxl'
# sentence_embedding = AutoPipes.pipeline('sentence_embedding', config=config)

# from sentence_transformers import SentenceTransformer
# sentence_embedding = SentenceTransformer('sentence-transformers/sentence-t5-xxl')


batch_num = int(len(data) / 10)
# ebds = None
ebds = []
print('start embedding')
for i in range(batch_num + 10):
    end = min(len(data), (i + 1) * 10)
    embeddings = sentence_embedding.batch(data[i * 10: end])
    tmp = [e.get() for e in embeddings]
    ebds += tmp
    
    # embeddings = sentence_embedding.encode(data[i * 1: end])
    # tmp = embeddings
    # ebds = tmp if ebds is None else np.concatenate((ebds, tmp), axis=0)
    
    print(end)
ebds = np.array(ebds)




to_fvecs2('/home/hadoop/wzy/dataset/miracl-22-12-sentence-t5-xxl1m/miracl-22-12-sentence-t5-xxl1m_query.fvecs', ebds)
