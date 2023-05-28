#pragma once

#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include "matrix.h"
#include "utils.h"
#include "adsampling.h"
using namespace std;

namespace svd{
    unsigned int D = 960;
    unsigned int delta_d = 96; // dimension sampling for every delta_d dimensions.

    Matrix<float> svd_table, queries_svd;

    unsigned int cur_query_label;


    float dist_comp(const float& bsf, unsigned label){

        float * q = &queries_svd.data[cur_query_label * D];
        float* d = &svd_table.data[label * D];
        float dis = 0;

        int i = 0;
        float dif = 0;

        while(i < D){
            // It continues to sample additional delta_d dimensions. 
            int check = std::min(delta_d, D-i);
            i += check;
            for(int j = 1;j<=check;j++){
                float t = *d - *q;
                d ++;
                q ++;
                dis += t * t;  
            }
            // Hypothesis tesing
            if(dis >= bsf){
                break;
            }
        }
        // for(; i < D && dis < bsf; ++i){
        //     dif = q[i] - svd_v[i];
        //     dis += dif * dif;
        // }

#ifdef COUNT_DIMENSION            
    adsampling::tot_dimension += i;
    // adsampling::tot_comp_dim += i;
#endif

        return i >= D ? dis : -dis;
    }

    float dist_comp2(const float& bsf, const void *data){

        float * q = &queries_svd.data[cur_query_label * D];
        float * d = (float *) data;
        float dis = 0;

        int i = 0;
        float dif = 0;

        while(i < D){
            // It continues to sample additional delta_d dimensions. 
            int check = std::min(delta_d, D-i);
            i += check;
            for(int j = 1;j<=check;j++){
                float t = *d - *q;
                d ++;
                q ++;
                dis += t * t;  
            }
            // Hypothesis tesing
            if(dis >= bsf){
                break;
            }
        }
        // for(; i < D && dis < bsf; ++i){
        //     dif = q[i] - svd_v[i];
        //     dis += dif * dif;
        // }

#ifdef COUNT_DIMENSION            
    adsampling::tot_dimension += i;
    // adsampling::tot_comp_dim += i;
#endif

        return i >= D ? dis : -dis;
    }


    // struct HashParam
    // {
    //     // the value of a in S hash functions
    //     float** rndAs = nullptr;
    //     // the value of b in S hash functions
    //     // float* rndBs = nullptr;
    //     // 
    //     //float W = 0.0f;

    //     //float calHash(float* point, )
    // };

    // HashParam hashPar;
    // hnswlib::DISTFUNC<float> ipdistfunc_;


    // float* calHash(float* point)
    // {
    //     float* res = new float[lowdim];
    //     for (int i = 0; i < lowdim; i++) {
    //         res[i] = (ipdistfunc_(point, hashPar.rndAs[i], &D));
    //     }
    //     return res;
    // }

        // void setHash()
    // {
    //     hashPar.rndAs = new float* [lowdim];
    //     // hashPar.rndBs = new float[S];

    //     for (int i = 0; i < lowdim; i++) {
    //         hashPar.rndAs[i] = new float[D];
    //     }

    //     //std::mt19937 rng(int(std::time(0)));
    //     std::mt19937 rng(int(0));
    //     // std::uniform_real_distribution<float> ur(0, W);
    //     std::normal_distribution<float> nd;//nd is a norm random variable generator: mu=0, sigma=1
    //     for (int j = 0; j < lowdim; j++){
    //         for (int i = 0; i < D; i++){
    //             hashPar.rndAs[j][i] = (nd(rng));
    //         }
    //         // hashPar.rndBs[j] = (ur(rng));
    //     }

    // }


};