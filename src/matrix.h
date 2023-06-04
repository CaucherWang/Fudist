
#pragma once
#ifndef MATRIX_HPP_
#define MATRIX_HPP_
#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <cstring>
#include <assert.h>
#include "Eigen/Dense"

template <typename T>
class Matrix
{
private:

public:
    T* data;
    size_t n;
    size_t d;
    size_t M, Ks, sub_vec_len;

    Matrix(); // Default
    Matrix(char * data_file_path); // IO
    Matrix(char * data_file_path, bool from_fdat); // IO from fdat (pq codebook)
    Matrix(size_t n, size_t d); // Matrix of size n * d

    // Deconstruction
    ~Matrix(){ delete [] data;}

    Matrix & operator = (const Matrix &X){
        delete [] data;
        n = X.n;
        d = X.d;
        data = new T [n*d];
        memcpy(data, X.data, sizeof(T) * n * d);
        return *this;
    }
    
    void mul(const Matrix<T> &A, Matrix<T> &result) const;
    float dist(size_t a, const Matrix<T> &B, size_t b) const;
    T* operator[](int row){
        return data + (row * d);
    }
};

template <typename T>
Matrix<T>::Matrix(){
    n = 0;
    d = 0;
    data = NULL;
}

template <typename T>
Matrix<T>::Matrix(char *data_file_path){
    n = 0;
    d = 0;
    data = NULL;
    std::cerr << data_file_path << std::endl;
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&d, 4);
    
    std::cerr << "Dimensionality - " <<  d <<std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    n = (size_t)(fsize / (d + 1) / 4);
    data = new T [(size_t)n * (size_t)d];
    std::cerr << "Cardinality - " << n << std::endl;
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * d), d * 4);
    }
    in.close();
}

template <typename T>
Matrix<T>::Matrix(char *data_file_path, bool from_fdat){
    M = 0; Ks = 0; sub_vec_len = 0;
    data = NULL;
    std::cerr << data_file_path << std::endl;
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&M, 4);
    in.read((char*)&Ks, 4);
    in.read((char*)&sub_vec_len, 4);
    
    std::cerr << M << " segments (sub-codebooks), "<< Ks << " centroids in each segment (sub-codebook), " 
    <<  sub_vec_len << " len of each segment" <<std::endl;
    // in.seekg(0, std::ios::end);
    // std::ios::pos_type ss = in.tellg();
    // size_t fsize = (size_t)ss;
    // n = (size_t)(fsize / (d + 1) / 4);
    // data = new T [(size_t)n * (size_t)d];
    // std::cerr << "Cardinality - " << n << std::endl;
    data = new T [(size_t)M * (size_t)Ks * (size_t)sub_vec_len];
    in.seekg(12, std::ios::beg);
    in.read((char*)(data), M * Ks * sub_vec_len * 4);
    // for (size_t i = 0; i < (size_t)M * (size_t)Ks; i++) {
    //     in.seekg(4, std::ios::cur);
    //     in.read((char*)(data + i * sub), d * 4);
    // }
    in.close();
}

template <typename T>
Matrix<T>::Matrix(size_t _n,size_t _d){
    n = _n;
    d = _d;
    data = new T [n * d];
}

template<typename T>
Matrix<T> mul(const Matrix<T> &A, const Matrix<T> &B){

    Eigen::MatrixXf _A(A.n, A.d);
    Eigen::MatrixXf _B(B.n, B.d);
    Eigen::MatrixXf _C(A.n, B.d);

    for(int i=0;i<A.n;i++)
        for(int j=0;j<A.d;j++)
            _A(i,j)=A.data[i*A.d+j];
    
    for(int i=0;i<B.n;i++)
        for(int j=0;j<B.d;j++)
            _B(i,j)=B.data[i*B.d+j];

    _C = _A * _B;

    Matrix<T> result(A.n, B.d);

    for(int i=0;i<A.n;i++)
        for(int j=0;j<B.d;j++)
            result.data[i*B.d+j] = _C(i,j);
    
    return result;
}

template<typename T>
float* vec_len(const Matrix<T> &A){

    Eigen::MatrixXf _A(A.n, A.d);

    float* result = new float[A.n];

    for(int i=0;i<A.n;i++)
        for(int j=0;j<A.d;j++)
            _A(i,j)=A.data[i*A.d+j];
    
    for(int i = 0; i < A.n; i++){
        result[i] = _A.row(i).squaredNorm();
    }
    
    return result;
}

template<typename T>
Matrix<float> to_paa(const Matrix<T> &A, unsigned seg_num){

    Matrix<float> result(A.n, seg_num);

    unsigned len_per_seg = A.d / seg_num;

    int s, i;
    float* ele;
    float* vec;
    for(int i =0 ;i < A.n; i++){  
        vec = A.data + i * A.d;   
        for (s=0; s<seg_num; s++) {
            ele = result.data + i * seg_num + s;
            *ele = 0;
            for (int j=0; j < len_per_seg; j++) {
                *ele += vec[(s * len_per_seg)+j];
            }
            *ele /= len_per_seg;
        }

    }
    
    return result;
}

template<typename T>
Matrix<float> to_paa_positions(const Matrix<T> &A, unsigned seg_num, unsigned* order){

    Matrix<float> result(A.n, seg_num);

    unsigned len_per_seg = A.d / seg_num;

    int s, i;
    float* ele;
    float* vec;
    for(int i =0 ;i < A.n; i++){  
        vec = A.data + i * A.d;   
        for (s=0; s<seg_num; s++) {
            ele = result.data + i * seg_num + s;
            *ele = 0;
            for (int j=0; j < len_per_seg; j++) {
                *ele += vec[order[(s * len_per_seg)+j]];
            }
            *ele /= len_per_seg;
        }

    }
    
    return result;
}

template <typename T>
float Matrix<T>::dist(size_t a, const Matrix<T> &B, size_t b)const{
    float dist = 0;
    for(size_t i=0;i<d;i++){
        dist += (data[a * d + i] - B.data[b * d + i]) * (data[a * d + i] - B.data[b * d + i]);
    }   
    return dist;
}

#endif
