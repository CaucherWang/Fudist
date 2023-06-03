
cd ..

efConstruction=500
M=48
data='word2vec'

g++ -o ./src/index_hnsw ./src/index_hnsw.cpp -I ./src/ -O3 -mavx2 -fopenmp


echo "Indexing - ${data}"

data_path=./data/${data}
index_path=./data/${data}

data_file="${data_path}/${data}_base.fvecs"
index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index"
./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

data_file="${data_path}/O${data}_base.fvecs"
index_file="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

data_file="${data_path}/PCA_${data}_base.fvecs"
index_file="${index_path}/PCA_${data}_ef${efConstruction}_M${M}.index"
./src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M
