cd ..

g++ ./src/search_ivf.cpp -O3 -o ./src/search_ivf -I ./src/ -ffast-math -march=native

cd src
./search_ivf
