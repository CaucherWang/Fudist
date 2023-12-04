import random

from n2 import HnswIndex

f = 40
t = HnswIndex(f)  # HnswIndex(f, "angular, L2, or dot")
for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_data(v)

t.build(m=5, max_m0=10, n_threads=4)
t.save('test.hnsw')

u = HnswIndex(f)
u.load('test.hnsw')
print(u.search_by_id(0, 1000))