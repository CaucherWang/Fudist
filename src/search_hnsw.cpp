#define USE_SIMD
// #define DEEP_DIVE
#define DEEP_QUERY

// #define COUNT_DIMENSION
// #define COUNT_FN
// #define COUNT_DIST_TIME
// #define ED2IP
// #define  SYNTHETIC
#ifndef USE_SIMD
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#endif


#include <iostream>
#include <fstream>
#include <gperftools/profiler.h>
#include <ctime>
#include <cmath>
#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"
#include "adsampling.h"
#include "paa.h"
#include "dwt.h"
#include "finger.h"
#include "seanet.h"
#include "svd.h"
#include "lsh.h"
#include "pq.h"

#include <getopt.h>

using namespace std;
using namespace hnswlib;

const int MAXK = 100;

long double rotation_time=0;

float *read_from_floats(const char *data_file_path){
    float * data = NULL;
    std::cerr << "Reading "<< data_file_path << std::endl;
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    size_t n = (size_t)(fsize / 4);
    // data = new T [(size_t)n * (size_t)d];
    // std::cerr << "Cardinality - " << n << std::endl;
    data = new float [(size_t)n];
    in.seekg(0, std::ios::beg);
    in.read((char*)(data), fsize);
    // for (size_t i = 0; i < (size_t)M * (size_t)Ks; i++) {
    //     in.seekg(4, std::ios::cur);
    //     in.read((char*)(data + i * sub), d * 4);
    // }
    in.close();
    return data;
}

static void get_gt(unsigned int *massQA, float *massQ, size_t vecsize, size_t qsize, SpaceInterface<float> &l2space,
       size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, size_t subk, HierarchicalNSW<float> &appr_alg) {

    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < subk; j++) {
            answers[i].emplace(appr_alg.fstdistfunc_(massQ + i * vecdim, appr_alg.getDataByInternalId(appr_alg.getInternalId(massQA[k * i + j])), appr_alg.dist_func_param_), massQA[k * i + j]);
        }
    }
}


int recall(std::priority_queue<std::pair<float, labeltype >> &result, std::priority_queue<std::pair<float, labeltype >> &gt){
    unordered_set<labeltype> g;
    while (gt.size()) {
        g.insert(gt.top().second);
        gt.pop();
    }

    int ret = 0;
    while (result.size()) {
        if (g.find(result.top().second) != g.end()) {
            ret++;
        }
        result.pop();
    }    
    return ret;
}


int recall_deep_dive(std::priority_queue<std::pair<float, labeltype >> &result, std::priority_queue<std::pair<float, labeltype >> &gt, HierarchicalNSW<float> &appr_alg, 
                    vector<int>&out_degree, vector<int>&in_degree){
    unordered_set<labeltype> g;
    vector<std::pair<float, labeltype >> gt_vec;
    vector<std::pair<float, labeltype >> result_vec;
    unordered_set<labeltype> r1, r2;
    while (gt.size()) {
        g.insert(gt.top().second);
        gt_vec.push_back(gt.top());
        r1.insert(gt.top().second);
        gt.pop();
    }
    assert(r1.size() == gt_vec.size());
    std::priority_queue<std::pair<float, labeltype >> results;
    while (result.size()) {
        result_vec.push_back(result.top());
        results.push(result.top());
        result.pop();
    }
    assert(results.size() == result_vec.size());

    int sum_outd = 0, sum_ind = 0;
    cout << "Ground Truth: Internal Id\t External Lable\t distance \t out_d \t in_d" << endl;
    for(int i = gt_vec.size() - 1; i >= 0; --i){
        cout << gt_vec.size() - i << ":\t" << appr_alg.getInternalId(gt_vec[i].second) << "\t" << gt_vec[i].second << '\t' << gt_vec[i].first 
        << '\t' << out_degree[appr_alg.getInternalId(gt_vec[i].second)] << '\t' << in_degree[appr_alg.getInternalId(gt_vec[i].second)] << endl;

        sum_outd += out_degree[appr_alg.getInternalId(gt_vec[i].second)];
        sum_ind += in_degree[appr_alg.getInternalId(gt_vec[i].second)];
    }
    cout << "Average out degree: " << (double)sum_outd / gt_vec.size() << endl;
    cout << "Average in degree: " << (double)sum_ind / gt_vec.size() << endl;

    sum_outd = 0, sum_ind = 0;

    cout << endl << "Result:" << endl;
    for(int i = result_vec.size() - 1; i >= 0; --i){
        cout << result_vec.size() - i << ":\t" << appr_alg.getInternalId(result_vec[i].second) << "\t" << result_vec[i].second << '\t' << result_vec[i].first 
        << '\t' << out_degree[appr_alg.getInternalId(result_vec[i].second)] << '\t' << in_degree[appr_alg.getInternalId(result_vec[i].second)] << endl;

        sum_outd += out_degree[appr_alg.getInternalId(result_vec[i].second)];
        sum_ind += in_degree[appr_alg.getInternalId(result_vec[i].second)];
    }
    cout << "Average out degree: " << (double)sum_outd / result_vec.size() << endl;
    cout << "Average in degree: " << (double)sum_ind / result_vec.size() << endl;
    cout << endl;

    // cout << "Ground Truth Vectors:" << endl;
    // for(int i = gt_vec.size() - 1; i >= 0; --i){
    //     auto vec = (float*)appr_alg.getDataByInternalId(appr_alg.getInternalId(gt_vec[i].second));
    //     cout << gt_vec.size() - i << ": ";
    //     for(int j =0; j < *(int*)appr_alg.dist_func_param_; ++j){
    //         cout << vec[j] << ",";
    //     }
    //     cout << endl;
    // }

    // cout << endl;

    // cout << "Result Vectors:" << endl;
    // for(int i = result_vec.size() - 1; i >= 0; --i){
    //     auto vec = (float*)appr_alg.getDataByInternalId(appr_alg.getInternalId(result_vec[i].second));
    //     cout << result_vec.size() - i << ": ";
    //     for(int j =0; j < *(int*)appr_alg.dist_func_param_; ++j){
    //         cout << vec[j] << ",";
    //     }
    //     cout << endl;
    // }

    int ret = 0;
    while (results.size()) {
        if (g.find(results.top().second) != g.end()) {
            ret++;
        }
        results.pop();
    }    
    return ret;
}

static void test_approx_deep_dive(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    adsampling::clear();

    vector<int>out_degree(appr_alg.cur_element_count, 0);
    vector<int>in_degree(appr_alg.cur_element_count, 0);
    appr_alg.getDegrees(out_degree, in_degree);

    for(int _ = 0; _ < 1; ++_){
        for (int i = 0; i < qsize; i++) {
            if(i != 0)  continue;
            paa::cur_query_label = i;
            lsh::cur_query_label = i;
            adsampling::cur_query_label = i;
            svd::cur_query_label = i;
            pq::cur_query_label = i;
            seanet::cur_query_label = i;
#ifdef ED2IP
    hnswlib::cur_query_vec_len = hnswlib::query_vec_len[i];
#endif
#ifndef WIN32
            float sys_t, usr_t, usr_t_sum = 0;  
            struct rusage run_start, run_end;
            GetCurTime( &run_start);
#endif
            std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k, adaptive);  
#ifndef WIN32
            GetCurTime( &run_end);
            GetTime( &run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;
#endif
            if(_ == 0){
                std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
                total += gt.size();
                int tmp = recall_deep_dive(result, gt, appr_alg, out_degree, in_degree);
                cout << tmp << ",";
                correct += tmp;   
            }
        }
    }
    cout << endl;

    double sum_out_d = 0, sum_in_d = 0;
    for(int i = 0; i < appr_alg.cur_element_count; ++i){
        sum_out_d += out_degree[i];
        sum_in_d += in_degree[i];
    }
    sum_out_d /= appr_alg.cur_element_count;
    sum_in_d /= appr_alg.cur_element_count;
    
    sort(out_degree.begin(), out_degree.end());
    sort(in_degree.begin(), in_degree.end());
    cout << "Average out degree: " << sum_out_d << endl;
    cout << "Min out degree:" << out_degree[0] << endl;
    cout << "25p out degree:" << out_degree[out_degree.size() / 4] << endl;
    cout << "Median out degree: " << out_degree[out_degree.size() / 2] << endl;
    cout << "75p out degree:" << out_degree[out_degree.size() * 3 / 4] << endl;
    cout << "Max out degree:" << out_degree[out_degree.size() - 1] << endl;

    cout << "Average in degree: " << sum_in_d << endl;
    cout << "Min in degree:" << in_degree[0] << endl;
    cout << "25p in degree:" << in_degree[in_degree.size() / 4] << endl;
    cout << "Median in degree: " << in_degree[in_degree.size() / 2] << endl;
    cout << "75p in degree:" << in_degree[in_degree.size() * 3 / 4] << endl;
    cout << "Max in degree:" << in_degree[in_degree.size() - 1] << endl;


    long double time_us_per_query = total_time / qsize / 3.0 + rotation_time;
    long double dist_calc_time = adsampling::distance_time / qsize / 3.0;
    long double app_dist_calc_time = adsampling::approx_dist_time / qsize / 3.0;
    long double approx_dist_per_query = adsampling::tot_approx_dist / (double)qsize / 3.0;
    long double full_dist_per_query = adsampling::tot_full_dist / (double)qsize / 3.0;
    long double tot_dist_per_query = adsampling::tot_dist_calculation / (double)qsize / 3.0;
    long double tot_dim_per_query = adsampling::tot_dimension / (double)qsize / 3.0;
    double fn_ratio = adsampling::tot_fn / (double)adsampling::tot_approx_dist;
    long double recall = 1.0f * correct / total;

    cout << setprecision(6);
    
    // cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query << " " << adsampling::tot_dimension + adsampling::tot_full_dist * vecdim << endl;
    cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query 
    << " ||| full dist time: " << dist_calc_time << " ||| approx. dist time: " << app_dist_calc_time 
    << " ||| #full dists: " << full_dist_per_query << " ||| #approx. dist: " << approx_dist_per_query 
    << endl << "\t\t" 
    << " ||| # total dists: " << (long long) tot_dist_per_query 
#ifdef COUNT_DIMENSION   
    << " ||| total dimensions: "<< (long long)tot_dim_per_query
#endif
    // << (long double)adsampling::tot_full_dist / (long double)adsampling::tot_dist_calculation 
    << " ||| pruining ratio (vector-level): " <<  (1 - full_dist_per_query / tot_dist_per_query) * 100.0
#ifdef COUNT_DIMENSION
    << " ||| pruning ratio (dimension-level)" << (1 - tot_dim_per_query / (tot_dist_per_query * vecdim)) * 100.0
#endif
    << endl << "\t\t ||| preprocess time: " << rotation_time  
#ifdef COUNT_FN
    <<" ||| FALSE negative ratio = " << fn_ratio * 100.0
#endif
    << endl;
    return ;
}



static void test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    adsampling::clear();

    for(int _ = 0; _ < 1; ++_){
        for (int i = 0; i < qsize; i++) {
            if(i != 2)  continue;
            paa::cur_query_label = i;
            lsh::cur_query_label = i;
            adsampling::cur_query_label = i;
            svd::cur_query_label = i;
            pq::cur_query_label = i;
            seanet::cur_query_label = i;
#ifdef ED2IP
    hnswlib::cur_query_vec_len = hnswlib::query_vec_len[i];
#endif
#ifndef WIN32
            float sys_t, usr_t, usr_t_sum = 0;  
            struct rusage run_start, run_end;
            GetCurTime( &run_start);
#endif
            std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k, adaptive);  
#ifndef WIN32
            GetCurTime( &run_end);
            GetTime( &run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;
#endif
            if(_ == 0){
                std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
                total += gt.size();
                int tmp = recall(result, gt);
                cout << tmp << endl;
                correct += tmp;   
            }
        }
    }

    long double time_us_per_query = total_time / qsize / 3.0 + rotation_time;
    long double dist_calc_time = adsampling::distance_time / qsize / 3.0;
    long double app_dist_calc_time = adsampling::approx_dist_time / qsize / 3.0;
    long double approx_dist_per_query = adsampling::tot_approx_dist / (double)qsize / 3.0;
    long double full_dist_per_query = adsampling::tot_full_dist / (double)qsize / 3.0;
    long double tot_dist_per_query = adsampling::tot_dist_calculation / (double)qsize / 3.0;
    long double tot_dim_per_query = adsampling::tot_dimension / (double)qsize / 3.0;
    double fn_ratio = adsampling::tot_fn / (double)adsampling::tot_approx_dist;
    long double recall = 1.0f * correct / total;

    cout << setprecision(6);
    
    // cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query << " " << adsampling::tot_dimension + adsampling::tot_full_dist * vecdim << endl;
    cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query 
    << " ||| full dist time: " << dist_calc_time << " ||| approx. dist time: " << app_dist_calc_time 
    << " ||| #full dists: " << full_dist_per_query << " ||| #approx. dist: " << approx_dist_per_query 
    << endl << "\t\t" 
    << " ||| # total dists: " << (long long) tot_dist_per_query 
#ifdef COUNT_DIMENSION   
    << " ||| total dimensions: "<< (long long)tot_dim_per_query
#endif
    // << (long double)adsampling::tot_full_dist / (long double)adsampling::tot_dist_calculation 
    << " ||| pruining ratio (vector-level): " <<  (1 - full_dist_per_query / tot_dist_per_query) * 100.0
#ifdef COUNT_DIMENSION
    << " ||| pruning ratio (dimension-level)" << (1 - tot_dim_per_query / (tot_dist_per_query * vecdim)) * 100.0
#endif
    << endl << "\t\t ||| preprocess time: " << rotation_time  
#ifdef COUNT_FN
    <<" ||| FALSE negative ratio = " << fn_ratio * 100.0
#endif
    << endl;
    return ;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    // vector<size_t> efs{100, 200, 300, 400, 500, 600, 750, 1000, 1500, 2000};
    // vector<size_t> efs{30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 400, 500, 600};
    // vector<size_t> efs{30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250};
    // vector<size_t> efs{60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 400, 500};
    // vector<size_t> efs{500, 600, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000};
    // vector<size_t> efs{300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000};
    // vector<size_t> efs{7000, 8000, 9000};
    // vector<size_t> efs{200, 250, 300, 400, 500, 600, 750, 1000, 1500};
    // vector<size_t> efs{100, 150, 200, 250, 300, 400, 500, 600};
    // vector<size_t> efs{1000,2000, 3000, 4000, 5000, 6000};
    vector<size_t> efs{50};
    // vector<size_t> efs{10000, 12500, 15000, 20000};
    // vector<size_t> efs{50, 100, 150};

        // ProfilerStart("../prof/svd-profile.prof");
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        #ifndef DEEP_DIVE
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive);
        #else
        test_approx_deep_dive(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive);
        #endif
    }
        // ProfilerStop();
}

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Query Parameter 
        {"randomized",                  required_argument, 0, 'd'},
        {"k",                           required_argument, 0, 'k'},
        {"epsilon0",                    required_argument, 0, 'e'},
        {"gap",                         required_argument, 0, 'p'},

        // Indexing Path 
        {"dataset",                     required_argument, 0, 'n'},
        {"index_path",                  required_argument, 0, 'i'},
        {"query_path",                  required_argument, 0, 'q'},
        {"groundtruth_path",            required_argument, 0, 'g'},
        {"result_path",                 required_argument, 0, 'r'},
        {"transformation_path",         required_argument, 0, 't'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    // 0: original HNSW,         2: ADS 3: PAA 4: LSH 5: SVD 6: PQ 7: OPQ 8: PCA 9:DWT 10:Finger 11:SEANet
    //                           20:ADS-keep        50: SVD-keep        80: PCA-keep
    //                           1: ADS+       41:LSH+             71: OPQ+ 81:PCA+       TMA optimize (from ADSampling)
    //                                                       62:PQ! 72:OPQ!              QEO optimize (from tau-MNG)
    int randomize = 0;
    string data_str = "deep";   // dataset name
    string M_str ="16"; // 8 for msong,mnist,cifar  48 for nuswide

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg)randomize = atoi(optarg);
                break;
        }
    }

#ifdef SYNTHETIC
    string syn_dim  = "50";
#endif
    float ads_epsilon0 = 2.1;
    int ads_delta_d = 16;
    int pca_delta_d = 16;
    int dwt_delta_d = 16;
    int paa_segment = 96;
    int lsh_dim = 16; 
    double lsh_p_tau = 0.95;
    int pq_m = 6;   // glove-100:4, msong, imagenet,word2vec:6  nuswide:10
    int pq_ks = 256;
    float pq_epsilon = 1;
    float qeo_check_threshold = 0.95;
    int qeo_check_num = 2;
    int finger_lsh_dim = 64;
    float finger_ratio = 1.5;
    float seanet_ratio = 1.5;

    int subk=20;
    string base_path_str = "../data";
    string result_base_path_str = "../results";
    string ef_str = "500"; 
    string exp_name = "";
    switch(randomize){
        case 0:
            break;
        case 1:
            exp_name = "ADS+";
            break;
        case 2:
            exp_name = "ADS";
            break;
        case 3:
            exp_name = "PAA";
            break;
        case 4:
            exp_name = "LSH" + to_string(lsh_dim);
            break;
        case 7:
            exp_name = "OPQ" + to_string(pq_m) + "-" + to_string(pq_ks);
            break;
        case 8:
            exp_name = "PCA";
            break;
        case 9:
            exp_name = "DWT";
            break;
        case 10:
            exp_name = "FINGER";
            break;
        case 11:
            exp_name = "SEANet";
            break;
    }

    string index_path_str = base_path_str + "/" + data_str + "/" + data_str + "_ef" + ef_str + "_M" + M_str + ".index";
    string ADS_index_path_str = base_path_str + "/" + data_str + "/O" + data_str + "_ef" + ef_str + "_M" + M_str + ".index";
    string PCA_index_path_str = base_path_str + "/" + data_str + "/PCA_" + data_str + "_ef" + ef_str + "_M" + M_str + ".index";
    string SVD_index_path_str = base_path_str + "/" + data_str + "/SVD_" + data_str + "_ef" + ef_str + "_M" + M_str + ".index";
    string DWT_index_path_str = base_path_str + "/" + data_str + "/DWT_" + data_str + "_ef" + ef_str + "_M" + M_str + ".index";
    string SEANet_index_path_str = base_path_str + "/SEANet_data/SEANet_" + data_str + "_ef" + ef_str + "_M" + M_str + ".index";
    string query_path_str = base_path_str + "/" + data_str + "/" + data_str + "_query.fvecs";
    string result_prefix_str = "";
    #ifdef USE_SIMD
    result_prefix_str += "SIMD_";
    #endif
    #ifdef ED2IP
    result_prefix_str += "IP_";
    #endif
    string result_path_str = result_base_path_str + "/" + data_str + "/" + result_prefix_str + data_str + "_ef" + ef_str + "_M" + M_str + "_" + exp_name + ".log";
    #ifdef DEEP_DIVE
    result_path_str += "_deepdive"; 
    #endif
    #ifdef DEEP_QUERY
    result_path_str += "_deepquery";
    #endif
    string groundtruth_path_str = base_path_str + "/" + data_str + "/" + data_str + "_groundtruth.ivecs";
    string trans_path_str = base_path_str + "/" + data_str + "/O.fvecs";
    string transed_data_path_str = base_path_str + "/" + data_str + "/O" + data_str + "_base.fvecs";
    string lsh_trans_path_str = base_path_str + "/" + data_str + "/LSH" + to_string(lsh_dim) + ".fvecs";
    string lsh_path_str = base_path_str + "/" + data_str + "/LSH" + to_string(lsh_dim) + "_" + data_str + "_base.fvecs";
    string svd_trans_path_str = base_path_str + "/" + data_str + "/SVD.fvecs";
    string pca_trans_path_str = base_path_str + "/" + data_str + "/PCA.fvecs";
    string svd_path_str = base_path_str + "/" + data_str + "/SVD_" + data_str + "_base.fvecs";
    string pca_path_str = base_path_str + "/" + data_str + "/PCA_" + data_str + "_base.fvecs";
    string pca_dist_distribution_path_str = base_path_str + "/" + data_str + "/PCA_dist_distrib.floats";
    string dwt_data_path_str = base_path_str + "/" + data_str + "/DWT_" + data_str + "_base.fvecs";
    string dwt_query_path_str = base_path_str + "/" + data_str + "/DWT_" + data_str + "_query.fvecs";
    string seanet_query_path_str = base_path_str + "/SEANet_data/SEANet_" + data_str + "_query.fvecs";
    string seanet_data_path_str = base_path_str + "/SEANet_data/SEANet_" + data_str + "_base.fvecs";
    string paa_path_str = base_path_str + "/" + data_str + "/PAA_" + to_string(paa_segment) + "_" + data_str + "_base.fvecs";
    string pq_codebook_path_str = base_path_str + "/" + data_str + "/PQ_codebook_" + to_string(pq_m) + "_" + to_string(pq_ks) + ".fdat";
    string pq_codes_path_str = base_path_str + "/" + data_str + "/PQ_" + to_string(pq_m) + "_" + to_string(pq_ks) + "_" + data_str + "_base.ivecs";
    string opq_rotation_path_str = base_path_str + "/" + data_str + "/OPQ_rotation_" + to_string(pq_m) + "_" + to_string(pq_ks)  + ".fvecs";
    string opq_codebook_path_str = base_path_str + "/" + data_str + "/OPQ_codebook_" + to_string(pq_m) + "_" + to_string(pq_ks) + ".fdat";
    string opq_codes_path_str = base_path_str + "/" + data_str + "/OPQ_" + to_string(pq_m) + "_" + to_string(pq_ks) + "_" + data_str + "_base.ivecs";
    string finger_projection_path_str = base_path_str + "/" + data_str + "/FINGER" + to_string(finger_lsh_dim) +"_" + data_str + "M" + M_str + "ef" + ef_str + "_LSH.fvecs";
    string finger_b_dres_path_str = base_path_str + "/" + data_str + "/FINGER" + to_string(finger_lsh_dim) +"_" + data_str + "M" + M_str + "ef" + ef_str +"_b_dres.fvecs";
    string finger_sgn_dres_P_path_str = base_path_str + "/" + data_str + "/FINGER" + to_string(finger_lsh_dim) +"_" + data_str + "M" + M_str + "ef" + ef_str +"_sgn_dres_P.ivecs";
    string finger_c_2_path_str = base_path_str + "/" + data_str + "/FINGER" + to_string(finger_lsh_dim) +"_" +  data_str + "M" + M_str + "ef" + ef_str +"_c_2.fvecs";
    string finger_c_P_path_str = base_path_str + "/" + data_str + "/FINGER" + to_string(finger_lsh_dim) +"_" +  data_str + "M" + M_str + "ef" + ef_str +"_c_P.fvecs";
    string finger_start_idx_path_str = base_path_str + "/" + data_str + "/FINGER" + to_string(finger_lsh_dim) +"_" +  data_str + "M" + M_str + "ef" + ef_str +"_start_idx.ivecs";
    
    char index_path[256];
    strcpy(index_path, index_path_str.c_str());
    char query_path[256] = "";
    strcpy(query_path, query_path_str.c_str());
    char groundtruth_path[256] = "";
    strcpy(groundtruth_path, groundtruth_path_str.c_str());
    char result_path[256];
    strcpy(result_path, result_path_str.c_str());
    char dataset[256] = "";
    strcpy(dataset, data_str.c_str());
    char transformation_path[256] = "";
    strcpy(transformation_path, trans_path_str.c_str());
    char transed_data_path[256] = "";
    strcpy(transed_data_path, transed_data_path_str.c_str());
    char paa_path[256] = "";
    strcpy(paa_path, paa_path_str.c_str());
    char lsh_trans_path[256] = "";
    strcpy(lsh_trans_path, lsh_trans_path_str.c_str());
    char lsh_path[256] = "";
    strcpy(lsh_path, lsh_path_str.c_str());
    char svd_trans_path[256] = "";
    strcpy(svd_trans_path, svd_trans_path_str.c_str());
    char pca_trans_path[256] = "";
    strcpy(pca_trans_path, pca_trans_path_str.c_str());
    char svd_path[256] = "";
    strcpy(svd_path, svd_path_str.c_str());
    char pca_path[256] = "";
    strcpy(pca_path, pca_path_str.c_str());
    char pq_codebook_path[256] = "";
    strcpy(pq_codebook_path, pq_codebook_path_str.c_str());
    char pq_codes_path[256] = "";
    strcpy(pq_codes_path, pq_codes_path_str.c_str());
    char opq_codebook_path[256] = "";
    strcpy(opq_codebook_path, opq_codebook_path_str.c_str());
    char opq_codes_path[256] = "";
    strcpy(opq_codes_path, opq_codes_path_str.c_str());
    char opq_rotation_path[256] = "";
    strcpy(opq_rotation_path, opq_rotation_path_str.c_str());
    char ads_index_path[256] = "";
    strcpy(ads_index_path, ADS_index_path_str.c_str());
    char svd_index_path[256] = "";
    strcpy(svd_index_path, SVD_index_path_str.c_str());
    char pca_index_path[256] = "";
    strcpy(pca_index_path, PCA_index_path_str.c_str());
    char dwt_index_path[256] = "";
    strcpy(dwt_index_path, DWT_index_path_str.c_str());
    char dwt_query_path[256] = "";
    strcpy(dwt_query_path, dwt_query_path_str.c_str());
    char seanet_index_path[256] = "";
    strcpy(seanet_index_path, SEANet_index_path_str.c_str());
    char seanet_query_path[256] = "";
    strcpy(seanet_query_path, seanet_query_path_str.c_str());
    char seanet_data_path[256] = "";
    strcpy(seanet_data_path, seanet_data_path_str.c_str());
    char finger_projection_path[256] = "";
    strcpy(finger_projection_path, finger_projection_path_str.c_str());
    char finger_b_dres_path[256] = "";
    strcpy(finger_b_dres_path, finger_b_dres_path_str.c_str());
    char finger_sgn_dres_P_path[256] = "";
    strcpy(finger_sgn_dres_P_path, finger_sgn_dres_P_path_str.c_str());
    char finger_c_2_path[256] = "";
    strcpy(finger_c_2_path, finger_c_2_path_str.c_str());
    char finger_c_P_path[256] = "";
    strcpy(finger_c_P_path, finger_c_P_path_str.c_str());
    char finger_start_idx_path[256] = "";
    strcpy(finger_start_idx_path, finger_start_idx_path_str.c_str());

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "i:q:g:r:t:n:k:e:p:", longopts, &ind);
        switch (iarg){
            // case 'd':
            //     if(optarg)randomize = atoi(optarg);
            //     break;
            case 'k':
                if(optarg)subk = atoi(optarg);
                break;
            case 'e':
                if(optarg)adsampling::epsilon0 = atof(optarg);
                break;
            case 'p':
                if(optarg)adsampling::delta_d = atoi(optarg);
                break;
            case 'i':
                if(optarg)strcpy(index_path, optarg);
                break;
            case 'q':
                if(optarg)strcpy(query_path, optarg);
                break;
            case 'g':
                if(optarg)strcpy(groundtruth_path, optarg);
                break;
            case 'r':
                if(optarg)strcpy(result_path, optarg);
                break;
            case 't':
                if(optarg)strcpy(transformation_path, optarg);
                break;
            case 'n':
                if(optarg)strcpy(dataset, optarg);
                break;
        }
    }
    
    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);

#ifdef ED2IP
    string base_vec_len_path = base_path_str + "/" + data_str + "/base_vec_len.floats";
    hnswlib::base_vec_len = read_from_floats(base_vec_len_path.c_str());
    hnswlib::query_vec_len = vec_len(Q);
    InnerProductSpace ip_space(Q.d);
    hnswlib::fstdistfuncIP = ip_space.get_dist_func();
#endif  

    cout << "result path: "<< result_path << endl;
    int simd_lowdim = -1;

    // adsampling::D = Q.d;
    freopen(result_path,"a",stdout);
    cout << "k: "<<subk << endl;;
    if(randomize == 1 || randomize == 2){
        cout << setprecision(2) << ads_epsilon0 << " " << ads_delta_d;
        if(randomize == 1) cout << " tight BSF ";
        cout << endl;
        adsampling::initialize(ads_delta_d);
        adsampling::D = Q.d;
        adsampling::epsilon0 = ads_epsilon0;
        adsampling::init_ratios();
        Matrix<float> P(transformation_path);
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        memset(index_path, 0, sizeof(index_path));
        strcpy(index_path, ads_index_path);
    }else if(randomize == 20){
        Matrix<float> P(transformation_path);
        adsampling::project_table = Matrix<float>(transed_data_path);
        StopW stopw = StopW();
        adsampling::queries_project = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
        adsampling::epsilon0 = ads_epsilon0;
        adsampling::delta_d = ads_delta_d;
        adsampling::init_ratios();
    }else if(randomize == 3){
        paa::paa_table = Matrix<float>(paa_path);
        StopW stopw = StopW();
        paa::queries_paa = to_paa(Q, paa_segment);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        paa::D = Q.d;
        paa::segment_num = paa_segment;
        paa::len_per_seg = paa::D / paa::segment_num;
    }else if(randomize == 4 || randomize == 41){
        cout << setprecision(2) << lsh_p_tau;
        if(randomize == 41) cout << "tight BSF";
        cout <<endl;
        Matrix<float> P(lsh_trans_path);
        lsh::lsh_table = Matrix<float>(lsh_path);
        StopW stopw = StopW();
        lsh::queries_lsh = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        lsh::D = Q.d;
        lsh::probQ = lsh_p_tau;
        lsh::lowdim = P.d;
        lsh::initialize();
    }else if(randomize == 50){
        Matrix<float> P(svd_trans_path);
        svd::svd_table = Matrix<float>(svd_path);
        StopW stopw = StopW();
        svd::queries_svd = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        svd::D = Q.d;
        svd::delta_d = pca_delta_d;
    }else if(randomize == 5){
        cout << pq_epsilon << endl;
        Matrix<float> P(svd_trans_path);
        // svd::svd_table = Matrix<float>(svd_path);
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        svd::D = Q.d;
        svd::delta_d = pca_delta_d;
        memset(index_path, 0, sizeof(index_path));
        strcpy(index_path, svd_index_path);
    } else if(randomize == 8 || randomize == 81){
        svd::D = Q.d;
        svd::delta_d = pca_delta_d;
        svd::initialize(pca_delta_d);
        Matrix<float> P(pca_trans_path);
        cout << pca_delta_d << " ";
        if(randomize == 81) cout << " tight BSF";
        cout << endl;
        // svd::svd_table = Matrix<float>(svd_path);
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        memset(index_path, 0, sizeof(index_path));
        strcpy(index_path, pca_index_path);
        if(randomize == 81){
            auto ret = read_from_floats(pca_dist_distribution_path_str.c_str());
            for(int i = 0; i < Q.d; ++i){
                ret[i] = 1.0 / ret[i];
            }
            svd::amp_ratios = ret;
        }
    }else if(randomize == 80){
        Matrix<float> P(pca_trans_path);
        svd::svd_table = Matrix<float>(pca_path);
        StopW stopw = StopW();
        svd::queries_svd = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        svd::D = Q.d;
        svd::delta_d = pca_delta_d;
    }else if(randomize == 6 || randomize == 61 || randomize == 62){
        pq::M = pq_m;
        pq::Ks = pq_ks;
        pq::D = Q.d;
        pq::epsilon = pq_epsilon;
        pq::sub_vec_len = Q.d / pq_m;
        Matrix<float> tmp(pq_codebook_path, true);
        pq::init_codebook(tmp);
        pq::pq_codes_base = Matrix<int>(pq_codes_path);
        pq::pq_codes_base.M = pq::pq_codes_base.d;
        // svd::svd_table = Matrix<float>(svd_path);
        StopW stopw = StopW();
        pq::calc_dist_book(Q);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;   
        cout << pq_epsilon << " ";
        if(randomize == 62){
            cout << qeo_check_threshold << " " << qeo_check_num;
            pq::qeo_check_threshold = qeo_check_threshold;
            pq::qeo_check_num = qeo_check_num;
        }
        cout << endl;
    }else if(randomize == 7 || randomize == 71 || randomize == 72){
        pq::M = pq_m;
        pq::Ks = pq_ks;
        pq::D = Q.d;
        pq::epsilon = pq_epsilon;
        pq::sub_vec_len = Q.d / pq_m;
        Matrix<float> tmp(opq_codebook_path, true);
        pq::init_codebook(tmp);
        Matrix<float>Rotation(opq_rotation_path);
        pq::pq_codes_base = Matrix<int>(opq_codes_path);
        pq::pq_codes_base.M = pq::pq_codes_base.d;
        // svd::svd_table = Matrix<float>(svd_path);
        StopW stopw = StopW();
        auto rotQ = mul(Q, Rotation);
        pq::calc_dist_book(rotQ);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        cout << pq_epsilon << " ";
        if(randomize == 72){
            pq::qeo_check_threshold = qeo_check_threshold;
            pq::qeo_check_num = qeo_check_num;
            cout << qeo_check_threshold << " " << qeo_check_num;
        }
        cout << endl;
    } else if(randomize == 9){  // DWT
        cout << dwt_delta_d << " ";
        if(randomize == 91) cout << " tight BSF";
        cout << endl;
        dwt::initialize(dwt_delta_d);
        StopW stopw = StopW();
        // TODO: dwt for query should be done in C++
        Q = Matrix<float>(dwt_query_path);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        dwt::D = Q.d;
        // dwt::delta_d = svd_delta_d;
        memset(index_path, 0, sizeof(index_path));
        strcpy(index_path, dwt_index_path);
    } else if(randomize == 10){ // finger
        cout << finger_ratio << " ";
        if(randomize == 101) cout << " tight BSF";
        cout << endl;
        finger::P = Matrix<float>(finger_projection_path);
        finger::bs_dres = Matrix<float>(finger_b_dres_path);
        finger::c_2s = Matrix<float>(finger_c_2_path);
        finger::c_Ps = Matrix<float>(finger_c_P_path);
        finger::start_idxs = Matrix<int>(finger_start_idx_path);
        finger::ratio = finger_ratio;
        finger::D = Q.d;
        finger::lsh_dim = finger_lsh_dim;
        Matrix<int> sgn_d_res_Ps = Matrix<int>(finger_sgn_dres_P_path);
        unsigned int edge_num = sgn_d_res_Ps.n;
        finger::binary_sgn_d_res_Ps = vector<unsigned long long>();
        for(int i = 0; i < edge_num; i++){
            finger::binary_sgn_d_res_Ps.push_back(
                finger::get_binary_sgn_from_array(sgn_d_res_Ps[i])
            );
        }

        StopW stopw = StopW();
        finger::q_Ps = mul(Q, finger::P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
    } else if(randomize == 11){  // SEANet
        cout << seanet_ratio << endl;
        seanet::lsh_table = Matrix<float>(seanet_data_path);
        StopW stopw = StopW();
        seanet::queries_lsh = Matrix<float>(seanet_query_path);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        seanet::D = Q.d;
        seanet::lowdim = seanet::queries_lsh.d;
        seanet::initialize();
        seanet::ratio = seanet_ratio;
    }
    
    Q.n = Q.n > 500 ? 500 : Q.n;
    L2Space space(Q.d);   
    cerr << "L2 space" << endl;
    // InnerProductSpace space(Q.d);
    // cerr << "IP space" << endl;
    HierarchicalNSW<float> *appr_alg = new HierarchicalNSW<float>(&space, index_path, false);
    size_t k = G.d;

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;

    get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, space, Q.d, answers, k, subk, *appr_alg);

    // ProfilerStart("../prof/svd-profile.prof");
    test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, randomize);
    // ProfilerStop();

    return 0;
}
