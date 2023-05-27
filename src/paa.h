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

    unsigned int segment_num = 16;
    unsigned int len_per_seg = D / segment_num;
    Matrix<float> paa_table, queries_paa;

    unsigned int cur_query_label;

    float dist_comp(const float& bsf, unsigned label){

        float * q = &queries_paa.data[cur_query_label * segment_num];
        float* paa_v = &paa_table.data[label * segment_num];
        float dis = 0;

        for(int i = 0; i < segment_num; ++i){
            dis += (q[i] - paa_v[i]) * (q[i] - paa_v[i]);
        }

        dis *= (D / segment_num);

        return dis >= bsf ? -dis : dis;
    }

    float * paaFromVector(const float* vec){
        // Create PAA representation
        auto* paa = new float [segment_num];

        int s, i;
        for (s=0; s<segment_num; s++) {
            paa[s] = 0;
            for (i=0; i<len_per_seg; i++) {
                paa[s] += vec[(s * len_per_seg)+i];
            }
            paa[s] /= len_per_seg;
        }
        return paa;
    }

};