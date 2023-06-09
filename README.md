# [] Fudist: An efficient distance approximation tool to accelerate the search of approximate nearest neighbors

We benchmark the performance of the following algorithms w.r.t. the distance approximation part:
1. ADSampling
2. LSH-APG (LSH-pruning part)
3. PQ
4. OPQ
5. PCA
6. DWT

And we benchmark and combine many heuristic idea and engineering optimizations from SOTA papers.
After that, we propose Fudist, the best tool for distance approximation and pruning in ANN search.


## Prerequisites

* Eigen == 3.4.0
    1. Download the Eigen library from https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz.
    2. Unzip it and move the `Eigen` folder to `./src/`.
    

---
## Reproduction

The tested datasets are available at https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html. 

1. Download and preprocess the datasets. Detailed instructions can be found in `./data/README.md`.

2. Index the datasets. It could take several hours. 
    ```sh
    # Index HNSW/HNSW+/HNSW++
    ./script/index_hnsw.sh
    ```
3. Test the queries of the datasets. The results are generated in `./results/`. Detailed configurations can be found in `./script/README.md`.
    ```sh
    # Index HNSW/HNSW+/HNSW++
    ./script/search_hnsw.sh
    ```
