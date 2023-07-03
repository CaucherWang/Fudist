#pragma once

#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include "matrix.h"
#include "utils.h"
#include "adsampling.h"
using namespace std;

namespace pq{
    unsigned int D = 960;
    unsigned M, Ks, sub_vec_len;
    Matrix<int> pq_codes_base;   // N * M
    // codebook: M * Ks * sub_vec_len, query_dist_book: n * M * Ks
    Matrix<float> codebook, query_dist_book;
    float epsilon = 1.1;
    float qeo_check_threshold;
    int qeo_check_num;

    unsigned int cur_query_label;

    L2Space* lowdimspace;
    hnswlib::DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_;


    static float
    L2sqr(const float *pVect1, const float *pVect2, size_t len);

    void init_codebook(Matrix<float>&tmp){
        codebook = Matrix<float>();
        codebook.M = M;
        codebook.Ks = Ks;
        codebook.sub_vec_len = sub_vec_len;
        codebook.data = new float[M * Ks * sub_vec_len];
        for(unsigned i = 0; i < M; i++){
            for(unsigned j = 0; j < Ks; j++){
                for(unsigned k = 0; k < sub_vec_len; k++){
                    codebook.data[i * Ks * sub_vec_len + j * sub_vec_len + k] = tmp.data[i * Ks * sub_vec_len + j * sub_vec_len + k];
                }
            }
        }
        lowdimspace = new L2Space(sub_vec_len);
        fstdistfunc_ = lowdimspace->get_dist_func();
        dist_func_param_ = lowdimspace->get_dist_func_param();

    }

    void calc_dist_book(const Matrix<float>&Q){
        // TODO: simd
        query_dist_book = Matrix<float>();
        query_dist_book.data = new float[Q.n * M * Ks];
        query_dist_book.n = Q.n;
        query_dist_book.M = M;
        query_dist_book.Ks = Ks;
        for(unsigned i = 0; i < Q.n; i++){
            for(unsigned j = 0; j < M; j++){
                for(unsigned k = 0; k < Ks; ++k){
                    query_dist_book.data[i * M * Ks + j * Ks + k] = fstdistfunc_(&Q.data[i * D + j * sub_vec_len], 
                                &codebook.data[j * Ks * sub_vec_len + k * sub_vec_len], dist_func_param_);
                }
            }
        }
    }


    float dist_comp(const float& bsf, unsigned label){
        int * cand = &pq_codes_base.data[label * M];

        float dis = 0;  
        int i = 0;
        auto start = cur_query_label * M * Ks;
        auto dt = query_dist_book.data + start;
        for(; i < M; ++i){
            dis += dt[i * Ks + cand[i]];
        }

        // buggy code
        // for(; i < M; i +=4){
        //     float dsim = 0;
        //     dsim += dt[*cand++];
        //     dt += Ks;
        //     dsim += dt[*cand++];
        //     dt += Ks;
        //     dsim += dt[*cand++];
        //     dt += Ks;
        //     dsim += dt[*cand++];
        //     dis += dsim;
        // }

        // for(; i < M; i +=4){
        //     float dsim = 0;
        //     dsim += dt[cand[i]];
        //     dt += Ks;
        //     dsim += dt[cand[i+1]];
        //     dt += Ks;
        //     dsim += dt[cand[i+2]];
        //     dt += Ks;
        //     dsim += dt[cand[i+3]];
        //     dis += dsim;
        // }
        #ifdef COUNT_DIMENSION            
    adsampling::tot_dimension += i;
    // adsampling::tot_comp_dim += i;
#endif

        return dis >= epsilon * bsf ? -dis : dis;
    }

    float dist_comp_naive(const float& bsf, unsigned label){
        int * cand = &pq_codes_base.data[label * M];

        float dis = 0;  
        int i = 0;
        for(; i < M; ++i){
            dis += query_dist_book.data[cur_query_label * M * Ks + i * Ks + cand[i]];
        }

#ifdef COUNT_DIMENSION            
    adsampling::tot_dimension += i;
    // adsampling::tot_comp_dim += i;
#endif

        return dis >= bsf ? -dis : dis;
    }

        static float
    L2sqr(const float *pVect1, const float *pVect2, size_t len) {

        float res = 0;
        for (size_t i = 0; i < len; i++) {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }


};