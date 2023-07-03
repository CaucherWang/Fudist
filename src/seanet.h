#pragma once


#include <cmath>
#include <limits>
#include <queue>
#include <vector>
#include <iostream>
#include "matrix.h"
#include "utils.h"
using namespace std;

namespace seanet{
    unsigned int D = 960;

    unsigned int lowdim; // number of hash functions 

    Matrix<float> lsh_table, queries_lsh;

    unsigned int cur_query_label;
    float amp, ratio;

    L2Space* lowdimspace;
    hnswlib::DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_;


    void initialize(){
        // ipdistfunc_ = s->get_dist_func();
        // setHash();

        lowdimspace = new L2Space(lowdim);
        fstdistfunc_ = lowdimspace->get_dist_func();
        dist_func_param_ = lowdimspace->get_dist_func_param();
        amp = D / lowdim;
	}


    float dist_comp(const float& bsf, unsigned label){

        float * q = queries_lsh.data + cur_query_label * lowdim;
        float* lsh_v = lsh_table.data + label * lowdim;
        float dis = 0;
        dis  = fstdistfunc_(q, lsh_v, dist_func_param_);
        // float tmp;
        // for(int i = 0; i < lowdim; i++){
        //     tmp = q[i] - lsh_v[i];
        //     dis += tmp * tmp;
        // }

        dis *= D / lowdim;
        // float t;

        // for(int i = 0; i < lowdim; ++i){
        //     t = *q - *lsh_v;
        //     q++;
        //     lsh_v++;
        //     dis += t * t;
        // }
#ifdef COUNT_DIMENSION
    adsampling::tot_dimension += lowdim;
#endif
        return dis >= bsf * ratio ? -dis : dis;
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