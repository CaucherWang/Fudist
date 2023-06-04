#pragma once

#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <math.h>
#include "matrix.h"
#include "utils.h"
#include "adsampling.h"
using namespace std;

namespace finger{
    unsigned int D = 960;
    unsigned int lsh_dim = 64; // number of hash functions 

    unsigned int q_label = 0;

    Matrix<float> P, c_2s, c_Ps, bs_dres, q_Ps;
    Matrix<int> start_idxs, sgn_d_res_Ps;

    vector<unsigned long long> binary_sgn_d_res_Ps;

    float p1, p2, p3, p4, cos_theta, dis;

    float calc_humming_distance(const unsigned long long& b1, const unsigned long long& b2){
        unsigned long long cur = 1, x = b1 ^ b2;
        float ret = 0;
        for(int i = 0; i < lsh_dim; i++){
            if(cur & x) ret += 1;
            cur <<= 1;
        }
        return ret / lsh_dim;
    }

    float dist_comp(const float& bsf, const float& t, const float& b, const float& c_2,
                    const float& d_res, const float& q_res, const float& q_res_2,
                    const unsigned long long& binary_sgn_q_res_P, const unsigned long long& binary_sgn_d_res_P){
        p1 = (t-b) * (t-b) * c_2;
        p2 = d_res * d_res;
        p3 = q_res_2;
        cos_theta = cos(calc_humming_distance(binary_sgn_q_res_P, binary_sgn_d_res_P) * M_PI);
        p4 = -2 * q_res * d_res * cos_theta;

        dis = p1 + p2 + p3 + p4;

#ifdef COUNT_DIMENSION            
adsampling::tot_dimension += lsh_dim + 16;
// adsampling::tot_comp_dim += i;
#endif

        return dis >= bsf ? dis : -dis;
    }

    unsigned long long get_binary_sgn_from_array(int * sgn_array){
        unsigned long long bi;
        for(int i = 0; i < lsh_dim; i++){
            bi <<= 1;
            bi |= sgn_array[i] > 0? 1: 0;
        }
        return bi;
    }

    

};