
cd ..

data=deep
efConstruction=500
M=16
recall=0.98
shuf=

# g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src
# g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src -march=native
# g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src -ffast-math -march=native
# g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src -ffast-math -march=native -fopenmp
# g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src -lprofiler
# path=./data/
# result_path=./results/


cd src
for i in {10..29}
do
    shuf="_shuf${i}"
    nohup ./search_hnsw -e ${efConstruction} -m ${M} -d ${data} -r ${recall} -s ${shuf} 2>&1 >> ${i}.out &
done
# ./search_hnsw -e ${efConstruction} -m ${M} -d ${data} -r ${recall} -s ${shuf}

echo "done"

# data='gist'
# ef=500
# M=16
# k=20

# for randomize in {0..2}
# do
# if [ $randomize == "1" ]
# then 
#     echo "HNSW++"
#     index="${path}/${data}/O${data}_ef${ef}_M${M}.index"
# elif [ $randomize == "2" ]
# then 
#     echo "HNSW+"
#     index="${path}/${data}/O${data}_ef${ef}_M${M}.index"
# else
#     echo "HNSW"
#     index="${path}/${data}/${data}_ef${ef}_M${M}.index"    
# fi

# res="${result_path}/${data}_ef${ef}_M${M}_${randomize}.log"
# query="${path}/${data}/${data}_query.fvecs"
# gnd="${path}/${data}/${data}_groundtruth.ivecs"
# trans="${path}/${data}/O.fvecs"

# ./src/search_hnsw -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${k}