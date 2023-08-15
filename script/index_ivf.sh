
cd ..
g++ -o ./src/index_ivf ./src/index_ivf.cpp -I ./src/ -O3 -mavx2 -fopenmp

C=4096
data='gist'

echo "Indexing - ${data}"

data_path=./data/${data}
index_path=./data/${data}

# raw vectors 
# data_file="${data_path}/${data}_base.fvecs"
# centroid_file="${data_path}/${data}_cn${C}.fvecs"
# index_file="${index_path}/${data}_ivf_cn${C}.index"

# preprocessed vectors
data_file="${data_path}/O${data}_base.fvecs"
centroid_file="${data_path}/O${data}_cn${C}.fvecs"
index_file="${index_path}/O${data}_ivf_cn${C}.index"

# 0 - IVF, 1 - IVF++, 2 - IVF+
# index_file="${index_path}/${data}_ivf_${C}_${adaptive}.index"

# ./src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive
./src/index_ivf -d $data_file -c $centroid_file -i $index_file
