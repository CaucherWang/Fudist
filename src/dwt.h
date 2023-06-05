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
    unsigned int delta_d = 96; // dimension sampling for every delta_d dimensions.

    L2Space* lowdimspace;
    hnswlib::DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_;

    void initialize(unsigned dd){
        delta_d = dd;
        lowdimspace = new L2Space(delta_d);
        fstdistfunc_ = lowdimspace->get_dist_func();
        dist_func_param_ = lowdimspace->get_dist_func_param();
    }


        float dist_comp_deltad(const float& bsf, const void *data, const void *query){
        float * q = (float *)query;
        float * d = (float *) data;
        float dis = 0;

        int i = 0;
        float dif = 0;

        while(i < D){
            // It continues to sample additional delta_d dimensions. 
            int check = std::min(delta_d, D-i);
            i += check;
            dis += fstdistfunc_(q, d, dist_func_param_);
            d+= check;
            q+= check;
            // for(int j = 1;j<=check;j++){
            //     float t = *d - *q;
            //     d ++;
            //     q ++;
            //     dis += t * t;  
            // }
            // Hypothesis tesing
            if(dis >= bsf){
                break;
            }
        }
#ifdef COUNT_DIMENSION            
    adsampling::tot_dimension += i;
    // adsampling::tot_comp_dim += i;
#endif

        return i >= D ? dis : -dis;
    }

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