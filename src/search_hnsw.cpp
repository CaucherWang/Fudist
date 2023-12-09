#define USE_SIMD
// #define DEEP_DIVE
// #define DEEP_QUERY
// #define STAT_QUERY
// #define FOCUS_QUERY (9766)

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
#include <iomanip>
// #include <gperftools/profiler.h>
#include <ctime>
#include <map>
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

template<typename data_t, typename dist_t>
static void get_gt(unsigned int *massQA, data_t *massQ, size_t vecsize, size_t qsize, SpaceInterface<dist_t> &l2space,
       size_t vecdim, vector<std::priority_queue<std::pair<dist_t, labeltype >>> &answers, size_t k, size_t subk, HierarchicalNSW<dist_t> &appr_alg) {

    (vector<std::priority_queue<std::pair<dist_t, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<dist_t> fstdistfunc_ = l2space.get_dist_func();
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < subk; j++) {
            answers[i].emplace(appr_alg.fstdistfunc_(massQ + i * vecdim, appr_alg.getDataByInternalId(appr_alg.getInternalId(massQA[k * i + j])), appr_alg.dist_func_param_), massQA[k * i + j]);
        }
    }
}

template<typename dist_t>
int recall(std::priority_queue<std::pair<dist_t, labeltype >> &result, std::priority_queue<std::pair<dist_t, labeltype >> &gt){
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

template<typename dist_t>
int recall_deep_dive(std::priority_queue<std::pair<dist_t, labeltype >> &result, std::priority_queue<std::pair<dist_t, labeltype >> &gt, HierarchicalNSW<dist_t> &appr_alg, 
                    vector<int>&out_degree, vector<int>&in_degree){
    unordered_set<labeltype> g;
    vector<std::pair<dist_t, labeltype >> gt_vec;
    vector<std::pair<dist_t, labeltype >> result_vec;
    unordered_set<labeltype> r1, r2;
    while (gt.size()) {
        g.insert(gt.top().second);
        gt_vec.push_back(gt.top());
        r1.insert(gt.top().second);
        gt.pop();
    }
    assert(r1.size() == gt_vec.size());
    std::priority_queue<std::pair<dist_t, labeltype >> results;
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
    //     auto vec = (dist_t*)appr_alg.getDataByInternalId(appr_alg.getInternalId(gt_vec[i].second));
    //     cout << gt_vec.size() - i << ": ";
    //     for(int j =0; j < *(int*)appr_alg.dist_func_param_; ++j){
    //         cout << vec[j] << ",";
    //     }
    //     cout << endl;
    // }

    // cout << endl;

    // cout << "Result Vectors:" << endl;
    // for(int i = result_vec.size() - 1; i >= 0; --i){
    //     auto vec = (dist_t*)appr_alg.getDataByInternalId(appr_alg.getInternalId(result_vec[i].second));
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

template<typename data_t, typename dist_t>
static void test_approx_deep_dive(data_t *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<dist_t> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<dist_t, labeltype >>> &answers, size_t k, int adaptive) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    adsampling::clear();

    vector<int>out_degree(appr_alg.cur_element_count, 0);
    vector<int>in_degree(appr_alg.cur_element_count, 0);
    appr_alg.getDegrees(out_degree, in_degree);

    for(int _ = 0; _ < 1; ++_){
        for (int i = 0; i < qsize; i++) {
            if(i !=40)  continue;
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
            std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k, adaptive);  
#ifndef WIN32
            GetCurTime( &run_end);
            GetTime( &run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;
#endif
            if(_ == 0){
                std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
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

    std::cout << setprecision(6);
    
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



template<typename data_t, typename dist_t>
static void test_approx(data_t *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<dist_t> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<dist_t, labeltype >>> &answers, size_t k, int adaptive) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    int expr_round = 1;

    adsampling::clear();

    
#ifdef DEEP_QUERY
    hnswlib::indegrees_.resize(appr_alg.cur_element_count, 0);    // internal id
    for(int i = 0 ; i <appr_alg.cur_element_count; i++){
        int *data = (int *) appr_alg.get_linklist0(appr_alg.getInternalId(i));
        size_t size = appr_alg.getListCount((linklistsizeint*)data);
        data = data + 1;
        for(int j = 0; j < size; ++j){
            indegrees_[data[j]]++;
        }
    }
    #ifdef STAT_QUERY
    hop_cnt.resize(appr_alg.cur_element_count, 0);
    #endif
#endif

    vector<long> ndcs(qsize, 0);
    vector<int> recalls(qsize, 0);
    long accum_ndc = 0;
    for(int _ = 0; _ < expr_round; ++_){
        for (int i = 0; i < qsize; i++) {
            #ifdef DEEP_QUERY
            if(i != FOCUS_QUERY)  continue;
            #endif
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
#ifdef DEEP_QUERY
            std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
            std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnnPlainDEEP_QUERY(massQ + vecdim * i, k, gt);
#else
            std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnnPlain(massQ + vecdim * i, k, adaptive);  
#endif
#ifndef WIN32
            GetCurTime( &run_end);
            GetTime( &run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;
#endif
            if(_ == 0){
                std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
                total += gt.size();
                int tmp = recall(result, gt);
                cout << tmp << ",";
                ndcs[i] += (adsampling::tot_full_dist - accum_ndc);
                recalls[i] = tmp;
                accum_ndc = adsampling::tot_full_dist;
                #ifdef DEEP_QUERY
                #ifndef STAT_QUERY
                cout << tmp << endl;
                #endif
                #endif
                correct += tmp;   
            }
        }
    }

#ifdef STAT_QUERY
    // find the top 50 hop index and their hop counts, indegrees,
    vector<pair<int, int>> top_hop_cnt;
    for(int i = 0; i < appr_alg.cur_element_count; ++i){
        top_hop_cnt.push_back(make_pair(hop_cnt[i], i));
    }
    sort(top_hop_cnt.begin(), top_hop_cnt.end(), greater<pair<int, int>>());
    for(int i = 0; i < 50; ++i){
        cout << i << ": " << top_hop_cnt[i].first << " " << top_hop_cnt[i].second << " " 
        << indegrees_[top_hop_cnt[i].second] << endl;
    }
    string location = "../results/sift_hop_distances.fbin";
    std::ofstream outfile(location);
    for(auto v:hop_distance){
        outfile << v << endl;
    }
    outfile.close();
#endif

    auto tmp = double(expr_round);
    cout << endl;
    for(auto &_: ndcs)
        cout << _ << ","; 
    cout << endl;
    cout << setprecision(4);
    for(int i =0;i<ndcs.size();++i)
        cout << (double)recalls[i] / (double)ndcs[i] * 100.0 << ",";
    long double time_us_per_query = total_time / qsize / tmp  + rotation_time;
    long double dist_calc_time = adsampling::distance_time / qsize / tmp;
    long double app_dist_calc_time = adsampling::approx_dist_time / qsize / tmp;
    long double approx_dist_per_query = adsampling::tot_approx_dist / (double)qsize / tmp;
    long double full_dist_per_query = adsampling::tot_full_dist / (double)qsize / tmp;
    long double hop_per_query = adsampling::tot_hops / (double)qsize / tmp;
    long double tot_dist_per_query = adsampling::tot_dist_calculation / (double)qsize / tmp;
    long double tot_dim_per_query = adsampling::tot_dimension / (double)qsize / tmp;
    double fn_ratio = adsampling::tot_fn / (double)adsampling::tot_approx_dist;
    long double recall = 1.0f * correct / total;

    cout << setprecision(6);
    cout << endl;
    // cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query << " " << adsampling::tot_dimension + adsampling::tot_full_dist * vecdim << endl;
    cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query << " ||| nhops: " << hop_per_query
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

template<typename data_t, typename dist_t>
static void test_vs_recall(data_t *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<dist_t> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<dist_t, labeltype >>> &answers, size_t k, int adaptive) {
    vector<size_t> efs{60,80, 100, 150, 200, 300, 400, 500, 600, 750, 1000, 1500, 2000};
    // vector<size_t> efs{30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 400, 500, 600};
    // vector<size_t> efs{30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 300,400,500,600};
    // vector<size_t> efs{30,40,50,60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 400};
    // vector<size_t> efs{500, 600, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000};
    // vector<size_t> efs{500, 600, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 20000};
    // vector<size_t> efs{300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000};
    // vector<size_t> efs{7000, 8000, 9000};
    // vector<size_t> efs{200, 250, 300, 400, 500, 600, 750, 1000, 1500, 2000, 3000, 4000};
    // vector<size_t> efs{100, 150, 200, 250, 300, 400, 500, 600};
    // vector<size_t> efs{90,92,94,96,98,100, 102,104,106,108,110};
    // vector<size_t> efs{140,142,144,146,148,150, 152,154,156,158,160};
    // vector<size_t> efs{1000,2000, 3000, 4000, 5000, 6000, 7000};
    // vector<size_t> efs{300,400,500,600};
    // vector<size_t> efs{10000, 12500, 15000, 20000};
    // vector<size_t> efs{100};

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

template<typename data_t, typename dist_t>
static void test_lb_recall(data_t *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<dist_t> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<dist_t, labeltype >>> &answers, size_t k, int adaptive) {
    double low_recall = 0.84, high_recall = 0.98;
    int point_num = 8;
    vector<int>recall_targets(point_num);
    cout << "Targets: ";
    for(int i = 0; i < point_num; ++i){
        recall_targets[i] = k * ((high_recall - low_recall) / (point_num - 1) * i + low_recall);
        cout << recall_targets[i] << ",";
    }
    cout << endl;
    int ef_bound = 10000;

    vector<vector<int>> results(qsize, vector<int>(point_num, 0));

#pragma omp parallel for
    for(int i = 0; i < qsize; ++i){
#pragma omp critical
        if(i % 100 == 0)
            cerr << i << endl;
        // if(i != 280) continue;
        std::map<int, int> recall_records;  // recall -> ef
        std::map<int,std::pair<int, int>> recall_records2; // ef->(recall,ndc)
        for(int j=0; j < point_num; ++j){
            int target = recall_targets[j];
            int left = 1, right = ef_bound, tmp = -1;
            if(recall_records.size() > 0){
                if(recall_records.count(target) > 0){
                    int ef = recall_records[target];
                    int ndc = recall_records2[ef].second;
                    results[i][j] = ndc;
                    continue;
                }
                // find the first ef that is larger than the target
                auto it = recall_records.lower_bound(target);
                if(it != recall_records.end()){
                    right = it->second;
                }
                // find the first ef that is smaller than the target
                it = recall_records.upper_bound(target);
                if(it != recall_records.begin()){
                    --it;
                    left = it->second;
                }
            }
            left = min(left, right);
            right = max(left, right);

            int success = -1;
            while(left < right){
                int mid = (left + right) / 2;
                if(recall_records2.count(mid) > 0){
                    auto temp = recall_records2[mid];
                    tmp = temp.first;
                    adsampling::tot_full_dist = temp.second;
                }
                else{
                    adsampling::clear();
                    appr_alg.setEf(mid);
                    std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnnPlain(massQ + vecdim * i, k, adaptive);  
                    std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
                    tmp = recall(result, gt);
                    recall_records2[mid] = make_pair(tmp, adsampling::tot_full_dist);
                    if(recall_records.count(tmp) == 0)  recall_records[tmp] = mid;
                    else    recall_records[tmp] = min(recall_records[tmp], mid);

                }
                
                if(tmp < target){
                    left = mid + 1;
                }else{
                    success = adsampling::tot_full_dist;
                    if(right == mid)
                        break;
                    right = mid;
                }
            }
            if(success >= 0){
                results[i][j] = success;
            }else if(tmp < target){
                // use right as ef
                int mid = right;
                if(recall_records2.count(mid) <= 0){
                    if(mid == ef_bound)
                        results[i][j] = adsampling::tot_full_dist;
                    else{
                        cerr << " Error " << i << "-" << j << endl;
                        exit(-1);
                    }
                }else
                    results[i][j] = recall_records2[mid].second;
            }else{
                cerr << "Error" << i << "-" << j << endl;
                exit(-1);
            }
        }
    }

    for(int i = 0; i < point_num; ++i){
        for(int j = 0; j < qsize; ++j){
            cout << results[j][i] << ",";
        }
        cout << endl;
    }
}

template<typename data_t, typename dist_t>
static void test_performance(data_t *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<dist_t> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<dist_t, labeltype >>> &answers, size_t k, int adaptive) {
    double target_recall = 0.98;
    int lowk = ceil(k * target_recall);
    vector<int>ret(qsize, 0);

    int index = 0;
#pragma omp parallel for
    for(int i = 0; i < qsize; ++i){
        bool flag = false;
        // if(i !=3)
        //     continue;
        #pragma omp critical
        {
            if(++index % 100 == 0)
                cerr << index << " / " << qsize << endl;
        }

        int lowef = k, highef, curef, tmp, bound = 20000;
        long success = -1;
        Metric metric;


        for(int _ = 0; _ < 1 && !flag; ++_){
            lowef = 10; highef = bound;
            success = -1;
            while(lowef < highef){
                curef = (lowef + highef) / 2;
                adsampling::clear();
                metric.clear();

                std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnnPlain(massQ + vecdim * i, k, adaptive, &metric, curef);  

                std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
                tmp = recall(result, gt);
                if(tmp < lowk){
                    lowef = curef+1;
                }else{
                    success = metric.ndc;
                    if(highef == curef)
                        break;
                    highef = curef;
                }
            }
            if(success >= 0){
                // if(success == 0){
                //     cerr << i << endl;
                //     exit(-1);
                // }
                ret[i] = success;
                flag = true;
            }
            else if(tmp >= lowk){
                // if(metric.ndc == 0){
                //     cerr << i << endl;
                //     exit(-1);
                // }
                ret[i] = metric.ndc;
                flag = true;
            }
            // if(tmp > highk){
            //     // if(lowef > 50){
            //     //     cerr << i << endl;
            //     // }
            //     long large_ndc = adsampling::tot_full_dist;
            //     curef = lowef;
            //     adsampling::clear();
            //     appr_alg.setEf(curef);

            //     std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnnPlain(massQ + vecdim * i, k, adaptive);  
            //     std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
            //     tmp = recall(result, gt);
            //     if(tmp >= lowk){
            //         cout << adsampling::tot_full_dist << ",";
            //     }else{
            //         cout << large_ndc << ",";
            //     }
            //     flag = true;
            else if(tmp < lowk){
                long large_ndc = metric.ndc;
                curef = highef;
                adsampling::clear();
                metric.clear();

                std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnnPlain(massQ + vecdim * i, k, adaptive, &metric, curef);  
                std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
                tmp = recall(result, gt);
                if(tmp >= lowk){
                    // if(metric.ndc == 0){
                    //     cerr << i << endl;
                    //     exit(-1);
                    // }
                    ret[i] = metric.ndc;
                    flag = true;
                }else if(curef >= bound){
                    cerr << i << endl;
                    ret[i] = metric.ndc;
                    flag = true;
                }
            }
        }
        if(!flag){
            cerr << i << endl;
            exit(-1);
        }
    }

    for(int i = 0; i < qsize; ++i){
        assert(ret[i] > 0);
        cout << ret[i] << ",";
    }
    cout << endl;
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
    int method = 0;
    string data_str = "sift";   // dataset name
    int data_type = 0; // 0 for float, 1 for uint8, 2 for int8
    string M_str ="16"; // 8 for msong,mnist,cifar  48 for nuswide
    string ef_str = "500"; 

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg)method = atoi(optarg);
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
    int subk=50;
    string base_path_str = "../data";
    string result_base_path_str = "../results";
    
    string exp_name = "perform_variance0.98";
    string index_postfix = "_plain";
    string query_postfix = "";
    // string index_postfix = "";
    string shuf_postfix = "_shuf5";
    switch(method){
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

    string index_path_str = base_path_str + "/" + data_str + "/" + data_str + "_ef" + ef_str + "_M" + M_str + ".index" + index_postfix + shuf_postfix;
    string data_path_str = base_path_str + "/" + data_str + "/" + data_str + "_base.fvecs" + shuf_postfix;
    string ADS_index_path_str = base_path_str + "/" + data_str + "/O" + data_str + "_ef" + ef_str + "_M" + M_str + ".index";
    string PCA_index_path_str = base_path_str + "/" + data_str + "/PCA_" + data_str + "_ef" + ef_str + "_M" + M_str + ".index";
    string SVD_index_path_str = base_path_str + "/" + data_str + "/SVD_" + data_str + "_ef" + ef_str + "_M" + M_str + ".index";
    string DWT_index_path_str = base_path_str + "/" + data_str + "/DWT_" + data_str + "_ef" + ef_str + "_M" + M_str + ".index";
    string SEANet_index_path_str = base_path_str + "/SEANet_data/SEANet_" + data_str + "_ef" + ef_str + "_M" + M_str + ".index";
    string query_path_str_postfix;
    if(data_type == 0)  query_path_str_postfix = ".fbin";
    else if(data_type == 1) query_path_str_postfix = ".u8bin";
    else if(data_type == 2) query_path_str_postfix = ".i8bin";
    query_path_str_postfix = ".fvecs";
    string query_path_str = base_path_str + "/" + data_str + "/" + data_str + "_query" + query_path_str_postfix + query_postfix;
    // string query_path_str = data_path_str;
    string result_prefix_str = "";
    #ifdef USE_SIMD
    result_prefix_str += "SIMD_";
    #endif
    #ifdef ED2IP
    result_prefix_str += "IP_";
    #endif
    string result_path_str = result_base_path_str + "/" + data_str + "/" + result_prefix_str + data_str + "_ef" + ef_str + "_M" + M_str + "_" + exp_name + ".log" + index_postfix + shuf_postfix + query_postfix;
    #ifdef DEEP_DIVE
    result_path_str += "_deepdive"; 
    #endif
    #ifdef DEEP_QUERY
    result_path_str += "_deepquery";
    #endif
    string groundtruth_path_str = base_path_str + "/" + data_str + "/" + data_str + "_groundtruth.ivecs" + shuf_postfix + query_postfix;
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
            //     if(optarg)method = atoi(optarg);
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
    
    cout << "result path: "<< result_path << endl;
    int simd_lowdim = -1;

    // adsampling::D = Q.d;
    freopen(result_path,"a",stdout);
    cout << "k: "<<subk << endl;;
    cerr << "ground truth path: " << groundtruth_path << endl;
    Matrix<unsigned> G(groundtruth_path);
    size_t k = G.d;
    unsigned q_num = 100654660;

    if(data_type == 0){
        cerr << "query path: " << query_path << endl;
        Matrix<float> Q(query_path);
        
        Q.n = Q.n > q_num ? q_num : Q.n;
        cerr << "Query number = " << Q.n << endl;

#ifdef ED2IP
    string base_vec_len_path = base_path_str + "/" + data_str + "/base_vec_len.floats";
    hnswlib::base_vec_len = read_from_floats(base_vec_len_path.c_str());
    hnswlib::query_vec_len = vec_len(Q);
    InnerProductSpace ip_space(Q.d);
    hnswlib::fstdistfuncIP = ip_space.get_dist_func();
#endif  

        if(method == 1 || method == 2){
            cout << setprecision(2) << ads_epsilon0 << " " << ads_delta_d;
            if(method == 1) cout << " tight BSF ";
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
        }else if(method == 20){
            Matrix<float> P(transformation_path);
            adsampling::project_table = Matrix<float>(transed_data_path);
            StopW stopw = StopW();
            adsampling::queries_project = mul(Q, P);
            rotation_time = stopw.getElapsedTimeMicro() / Q.n;
            adsampling::D = Q.d;
            adsampling::epsilon0 = ads_epsilon0;
            adsampling::delta_d = ads_delta_d;
            adsampling::init_ratios();
        }else if(method == 3){
            paa::paa_table = Matrix<float>(paa_path);
            StopW stopw = StopW();
            paa::queries_paa = to_paa(Q, paa_segment);
            rotation_time = stopw.getElapsedTimeMicro() / Q.n;
            paa::D = Q.d;
            paa::segment_num = paa_segment;
            paa::len_per_seg = paa::D / paa::segment_num;
        }else if(method == 4 || method == 41){
            cout << setprecision(2) << lsh_p_tau;
            if(method == 41) cout << "tight BSF";
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
        }else if(method == 50){
            Matrix<float> P(svd_trans_path);
            svd::svd_table = Matrix<float>(svd_path);
            StopW stopw = StopW();
            svd::queries_svd = mul(Q, P);
            rotation_time = stopw.getElapsedTimeMicro() / Q.n;
            svd::D = Q.d;
            svd::delta_d = pca_delta_d;
        }else if(method == 5){
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
        } else if(method == 8 || method == 81){
            svd::D = Q.d;
            svd::delta_d = pca_delta_d;
            svd::initialize(pca_delta_d);
            Matrix<float> P(pca_trans_path);
            cout << pca_delta_d << " ";
            if(method == 81) cout << " tight BSF";
            cout << endl;
            // svd::svd_table = Matrix<float>(svd_path);
            StopW stopw = StopW();
            Q = mul(Q, P);
            rotation_time = stopw.getElapsedTimeMicro() / Q.n;
            memset(index_path, 0, sizeof(index_path));
            strcpy(index_path, pca_index_path);
            if(method == 81){
                auto ret = read_from_floats(pca_dist_distribution_path_str.c_str());
                for(int i = 0; i < Q.d; ++i){
                    ret[i] = 1.0 / ret[i];
                }
                svd::amp_ratios = ret;
            }
        }else if(method == 80){
            Matrix<float> P(pca_trans_path);
            svd::svd_table = Matrix<float>(pca_path);
            StopW stopw = StopW();
            svd::queries_svd = mul(Q, P);
            rotation_time = stopw.getElapsedTimeMicro() / Q.n;
            svd::D = Q.d;
            svd::delta_d = pca_delta_d;
        }else if(method == 6 || method == 61 || method == 62){
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
            if(method == 62){
                cout << qeo_check_threshold << " " << qeo_check_num;
                pq::qeo_check_threshold = qeo_check_threshold;
                pq::qeo_check_num = qeo_check_num;
            }
            cout << endl;
        }else if(method == 7 || method == 71 || method == 72){
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
            if(method == 72){
                pq::qeo_check_threshold = qeo_check_threshold;
                pq::qeo_check_num = qeo_check_num;
                cout << qeo_check_threshold << " " << qeo_check_num;
            }
            cout << endl;
        } else if(method == 9){  // DWT
            cout << dwt_delta_d << " ";
            if(method == 91) cout << " tight BSF";
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
        } else if(method == 10){ // finger
            cout << finger_ratio << " ";
            if(method == 101) cout << " tight BSF";
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
        } else if(method == 11){  // SEANet
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
       
        L2Space space(Q.d);   
        cerr << "L2 space" << endl;
        cerr << "Read index from " << index_path << endl;
        auto appr_alg = new HierarchicalNSW<float>(&space, index_path, false);
        cerr << "max level: " << appr_alg->maxlevel_ << endl;
        
        vector<std::priority_queue<std::pair<float, labeltype >>> answers;
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, space, Q.d, answers, k, subk, *appr_alg);
        // ProfilerStart("../prof/svd-profile.prof");
        // test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, method);
        test_performance(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, method);
        // test_lb_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, method);
        // ProfilerStop();

    }else if(data_type == 1){
        Matrix<uint8_t> Q(query_path);
        
        Q.n = Q.n > q_num ? q_num : Q.n;
        cerr << "Query number = " << Q.n << endl;

        L2SpaceU8 space(Q.d);
        std::cerr << "L2 space (UINT 8)" << endl;
        auto appr_alg = new HierarchicalNSW<int>(&space, index_path, false);
        cerr << "max level: " << appr_alg->maxlevel_ << endl;
        vector<std::priority_queue<std::pair<int, labeltype >>> answers;
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, space, Q.d, answers, k, subk, *appr_alg);
        // ProfilerStart("../prof/svd-profile.prof");
        test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, method);

    }else if(data_type == 2){
        Matrix<int8_t> Q(query_path);
        
        Q.n = Q.n > q_num ? q_num : Q.n;
        cerr << "Query number = " << Q.n << endl;

        L2SpaceI8 space(Q.d);
        std::cerr << "L2 space (INT 8)" << endl;
        auto appr_alg = new HierarchicalNSW<int>(&space, index_path, false);
        cerr << "max level: " << appr_alg->maxlevel_ << endl;
        vector<std::priority_queue<std::pair<int, labeltype >>> answers;
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, space, Q.d, answers, k, subk, *appr_alg);
        // ProfilerStart("../prof/svd-profile.prof");
        test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, method);

    }
    // InnerProductSpace space(Q.d);
    // cerr << "IP space" << endl;

    return 0;
}
