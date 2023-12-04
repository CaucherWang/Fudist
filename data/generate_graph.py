import os
import numpy as np
import re

def resolve_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        edges = []
        query_edges = []
        for line in lines:
            line = line.strip()
            len_line = len(line)
            line_str = line.strip()
            line = re.split(r'\t| ', line_str)
            if len_line < 20 or ((  line[0][0] == '|' or '|' in line_str or 'h' in line_str) and line_str[0] != '#'):
                continue
            if(line[0][0] == '#'):
                edges.clear()
                query_edges.clear()
                continue
            line = [x for x in line if x != '']
            line = [line[3]] + line[6:9]
            for i in range(len(line)):
                if i not in [0,2]:
                    line[i] = int(line[i])
                else:
                    line[i] = float(line[i])
            query_edges.append([line[-1], -1, line[0]])
            if line[1] != line[3]:
                edges.append([line[1], line[3], line[2]])
    return edges, query_edges


source = './results/'
dataset = 'deep'
ef = 500
M = 16
nq = 9766
path = os.path.join(source, dataset)
hnsw_path = os.path.join(path, f'SIMD_{dataset}_ef{ef}_M{M}_.log_plain_1th_deepquery')
edges, query_edges = resolve_log(hnsw_path)
# edges = np.array(edges)
# query_edges = np.array(query_edges)
# data = np.concatenate((edges, query_edges), axis=0)
data = edges

graph_file = os.path.join(path, f'{dataset}_ef{ef}_M{M}_Q{nq}_graph.csv')
column_titles = ['source', 'target', 'weight']
formats = ['%d', '%d', '%.4f']

print(f'save to {graph_file}')
np.savetxt(graph_file, data, delimiter=',', header=','.join(column_titles), comments='', fmt=formats)
