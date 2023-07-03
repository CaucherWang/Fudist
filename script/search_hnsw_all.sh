
cd ..
# g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src
g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src -march=native

cd src
./search_hnsw -d 0
./search_hnsw -d 2
./search_hnsw -d 4
./search_hnsw -d 7
./search_hnsw -d 8
./search_hnsw -d 9
./search_hnsw -d 10
echo "done"


