#pragma once


#include <cmath>
#include <limits>
#include <queue>
#include <vector>
#include <iostream>
#include "matrix.h"

using namespace std;

namespace paa{
    unsigned int D = 960;

    unsigned int S;
    unsigned int W;
    unsigned int lowdim;

    unsigned int cur_query_label;

    struct HashParam
    {
        // the value of a in S hash functions
        float** rndAs = nullptr;
        // the value of b in S hash functions
        float* rndBs = nullptr;
        // 
        //float W = 0.0f;

        //float calHash(float* point, )
    };

    HashParam hashpar;

    float dist_comp(const float& bsf, unsigned label){

        // float * q = &queries_paa.data[cur_query_label * segment_num];
        // float* paa_v = &paa_table.data[label * segment_num];
        // float dis = 0;

        // for(int i = 0; i < segment_num; ++i){
        //     dis += (q[i] - paa_v[i]) * (q[i] - paa_v[i]);
        // }

        // dis *= (D / segment_num);

        // return dis >= bsf ? -dis : dis;
    }

    // inline float cal_inner_product(float* v1, float* v2, int dim)
    // {
    // #if (defined __AVX2__ && defined __USE__AVX2__ZX__)
    //     return faiss::fvec_inner_product_avx512(v1, v2, dim);
    // #else
    //     return calIp_fast(v1, v2, dim);
    // #endif
    // }

    // float* calHash(float* point)
    // {
    //     float* res = new float[S];
    //     for (int i = 0; i < S; i++) {
    //         res[i] = (cal_inner_product(point, hashPar.rndAs[i], lowdim) + hashPar.rndBs[i]) / W;
    //     }
    //     return res;
    // }


};