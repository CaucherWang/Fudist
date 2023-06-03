#pragma once

#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include "matrix.h"
#include "utils.h"
#include "adsampling.h"
using namespace std;

namespace dwt{
    unsigned int D = 960;
    // unsigned int delta_d = 96; // dimension sampling for every delta_d dimensions.

    float dist_comp(const float& bsf, const void *data, const void *query){

        float * q = (float *)query;
        float * d = (float *) data;
        float dis = 0;

        int i = 0;
        float dif = 0;
        for(; i < D && dis < bsf; ++i){
            dif = q[i] - d[i];
            dis += dif * dif;
        }

#ifdef COUNT_DIMENSION            
adsampling::tot_dimension += i;
// adsampling::tot_comp_dim += i;
#endif

        return i >= D ? dis : -dis;
    }

};