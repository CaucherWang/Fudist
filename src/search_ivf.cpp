// #define USE_SIMD
#define COUNT_DIMENSION
// #define COUNT_DIST_TIME

#ifndef USE_SIMD
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#endif

#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <ivf/ivf.h>
#include "hnswlib/hnswlib.h"
#include <adsampling.h>
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

static void get_gt(unsigned int *massQA, float *massQ, size_t vecsize, size_t qsize,
       size_t vecdim, vector<ResultHeap> &answers, size_t k, size_t subk, const IVF &ivf) {

    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    L2Space l2space(ivf.D);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    void *dist_func_param_ = l2space.get_dist_func_param();

    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < subk; j++) {
            answers[i].emplace(fstdistfunc_(massQ + i * vecdim, ivf.L1_data + massQA[k * i + j] * vecdim, dist_func_param_), massQA[k * i + j]);
        }
    }
}


int recall(std::priority_queue<std::pair<float, labeltype >> &result, std::priority_queue<std::pair<float, labeltype >> &gt){
    unordered_set<labeltype> g;
    int ret = 0;
    while (gt.size()) {
        g.insert(gt.top().second);
        gt.pop();
    }
    while (result.size()) {
        if (g.find(result.top().second) != g.end()) {
            ret++;
        }
        result.pop();
    }    
    return ret;
}


static void test_approx(float *massQ, size_t vecsize, size_t qsize, const IVF &ivf, size_t vecdim,
            vector<ResultHeap> &answers, size_t k, int nprobe, int adaptive) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    adsampling::clear();

    for(int _ = 0; _ < 3; ++_){
        for (int i = 0; i < qsize; i++) {
            // paa::cur_query_label = i;
            // lsh::cur_query_label = i;
            adsampling::cur_query_label = i;
            // svd::cur_query_label = i;
            // pq::cur_query_label = i;
            // seanet::cur_query_label = i;
#ifdef ED2IP
    hnswlib::cur_query_vec_len = hnswlib::query_vec_len[i];
#endif
#ifndef WIN32
            float sys_t, usr_t, usr_t_sum = 0;  
            struct rusage run_start, run_end;
            GetCurTime( &run_start);
#endif
            ResultHeap result = ivf.search(massQ + vecdim * i, k, nprobe, adaptive); 
            // KNNs = ivf.search(Q.data + i * Q.d, k, nprobe, adaptive); 
#ifndef WIN32
            GetCurTime( &run_end);
            GetTime( &run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;
#endif
            if(_ == 0){
                std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
                total += gt.size();
                int tmp = recall(result, gt);
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
    cout << nprobe << " " << recall * 100.0 << " " << time_us_per_query 
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

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, const IVF &ivf, size_t vecdim,
               vector<ResultHeap> &answers, size_t k, int adaptive) {

    vector<int> nprobes{100};

    for (size_t nprobe : nprobes) {
        test_approx(massQ, vecsize, qsize, ivf, vecdim, answers, k, nprobe, adaptive);
    }
}


// void test(const Matrix<float> &Q, const Matrix<unsigned> &G, const IVF &ivf, int k){
//     float sys_t, usr_t, usr_t_sum = 0, total_time=0, search_time=0;
//     struct rusage run_start, run_end;

//     vector<int> nprobes;
//     nprobes.push_back(100);
    
//     for(auto nprobe:nprobes){
//         total_time=0;
//         adsampling::clear();
//         int correct = 0;

//         for(int i=0;i<Q.n;i++){
//             GetCurTime( &run_start);
//             ResultHeap KNNs = ivf.search(Q.data + i * Q.d, k, nprobe);
//             GetCurTime( &run_end);
//             GetTime(&run_start, &run_end, &usr_t, &sys_t);
//             total_time += usr_t * 1e6;
//             // Recall
//             while(KNNs.empty() == false){
//                 int id = KNNs.top().second;
//                 KNNs.pop();
//                 for(int j=0;j<k;j++)
//                     if(id == G.data[i * G.d + j])correct ++;
//             }
//         }
//         float time_us_per_query = total_time / Q.n + rotation_time;
//         float recall = 1.0f * correct / (Q.n * k);
        
//         // (Search Parameter, Recall, Average Time/Query(us), Total Dimensionality)
//         cout << nprobe << " " << recall * 100.00 << " " << time_us_per_query << " " << adsampling::tot_dimension << endl;
//     }
// }

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Query Parameter 
        {"randomized",                  required_argument, 0, 'd'},
        {"K",                           required_argument, 0, 'k'},
        {"epsilon0",                    required_argument, 0, 'e'},
        {"delta_d",                     required_argument, 0, 'p'},

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

    char index_path[256] = "";
    char query_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    char dataset[256] = "";
    char transformation_path[256] = "";

    int randomize = 0;
    string data_str = "gist";
    string cn_str = "4096";

    string result_prefix_str = "";
    #ifdef USE_SIMD
    result_prefix_str += "SIMD_";
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

    int subk = 20;

    string base_path_str = "../data";
    string result_base_path_str = "../results_ivf";
    string exp_name = "";

    switch(randomize){
        case 0:
            break;
        case 1:
            exp_name = "ADS";
            break;
        case 2:
            exp_name = "ADS-KEEP";
            break;
        // case 3:
        //     exp_name = "PAA";
        //     break;
        // case 4:
        //     exp_name = "LSH" + to_string(lsh_dim);
        //     break;
        // case 7:
        //     exp_name = "OPQ" + to_string(pq_m) + "-" + to_string(pq_ks);
        //     break;
        // case 8:
        //     exp_name = "PCA";
        //     break;
        // case 9:
        //     exp_name = "DWT";
        //     break;
        // case 10:
        //     exp_name = "FINGER";
        //     break;
        // case 11:
        //     exp_name = "SEANet";
        //     break;
    }

    // while(iarg != -1){
    //     iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:", longopts, &ind);
    //     switch (iarg){
    //         case 'd':
    //             if(optarg)randomize = atoi(optarg);
    //             break;
    //         case 'k':
    //             if(optarg)subk = atoi(optarg);
    //             break;  
    //         case 'e':
    //             if(optarg)adsampling::epsilon0 = atof(optarg);
    //             break;
    //         case 'p':
    //             if(optarg)adsampling::delta_d = atoi(optarg);
    //             break;              
    //         case 'i':
    //             if(optarg)strcpy(index_path, optarg);
    //             break;
    //         case 'q':
    //             if(optarg)strcpy(query_path, optarg);
    //             break;
    //         case 'g':
    //             if(optarg)strcpy(groundtruth_path, optarg);
    //             break;
    //         case 'r':
    //             if(optarg)strcpy(result_path, optarg);
    //             break;
    //         case 't':
    //             if(optarg)strcpy(transformation_path, optarg);
    //             break;
    //         case 'n':
    //             if(optarg)strcpy(dataset, optarg);
    //             break;
    //     }
    // }

    // default settings
    string index_path_str = base_path_str + "/" + data_str + "/" + data_str + "_cn" + cn_str + ".index";
    string query_path_str = base_path_str + "/" + data_str + "/" + data_str + "_query.fvecs";
    string groundtruth_path_str = base_path_str + "/" + data_str + "/" + data_str + "_groundtruth.ivecs";
    string result_path_str = result_base_path_str + "/" + data_str + "/" + result_prefix_str + data_str + "_cn" + cn_str + "_" + exp_name + ".log";

    char index_path[256];
    strcpy(index_path, index_path_str.c_str());
    char query_path[256] = "";
    strcpy(query_path, query_path_str.c_str());
    char groundtruth_path[256] = "";
    strcpy(groundtruth_path, groundtruth_path_str.c_str());
    char result_path[256];
    strcpy(result_path, result_path_str.c_str());

    // adsampling
    string ADS_index_path_str = base_path_str + "/" + data_str + "/O" + data_str + "_cn" + cn_str + ".index";
    string trans_path_str = base_path_str + "/" + data_str + "/O.fvecs";

    char ads_index_path[256] = "";
    strcpy(ads_index_path, ADS_index_path_str.c_str());
    char transformation_path[256] = "";
    strcpy(transformation_path, trans_path_str.c_str());



    freopen(result_path,"a",stdout);

    // default data loading
    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);


    if(randomize == 1){
        cout << setprecision(2) << ads_epsilon0 << " " << ads_delta_d;
        // if(randomize == 1) cout << " tight BSF ";
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
    }
    
    Q.n = Q.n > 500 ? 500 : Q.n;
    IVF ivf;
    ivf.load(index_path);

    size_t k = G.d;

    vector<ResultHeap> answers;

    get_gt(G.data, Q.data, ivf.D, Q.n, Q.d, answers, k, subk, ivf);
    // ProfilerStart("../prof/svd-profile.prof");
    test_vs_recall(Q.data, ivf.D, Q.n, ivf, Q.d, answers, subk, randomize);
    // ProfilerStop();

    // test(Q, G, ivf, subk);
    return 0;
}
