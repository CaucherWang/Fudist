

// #define EIGEN_DONT_PARALLELIZE
// #define EIGEN_DONT_VECTORIZE
#define COUNT_DIMENSION
// #define COUNT_DIST_TIME

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
#include "svd.h"
#include "lsh.h"
#include "pq.h"

#include <getopt.h>

using namespace std;
using namespace hnswlib;

const int MAXK = 100;

long double rotation_time=0;

static void get_gt(unsigned int *massQA, float *massQ, size_t vecsize, size_t qsize, L2Space &l2space,
       size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, size_t subk, HierarchicalNSW<float> &appr_alg) {

    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < subk; j++) {
            answers[i].emplace(appr_alg.fstdistfunc_(massQ + i * vecdim, appr_alg.getDataByInternalId(massQA[k * i + j]), appr_alg.dist_func_param_), massQA[k * i + j]);
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


static void test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    adsampling::clear();

    for(int _ = 0; _ < 3; ++_){
        for (int i = 0; i < qsize; i++) {
            paa::cur_query_label = i;
            lsh::cur_query_label = i;
            adsampling::cur_query_label = i;
            svd::cur_query_label = i;
            pq::cur_query_label = i;
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
                correct += tmp;   
            }
        }
    }

    long double time_us_per_query = total_time / qsize / 3.0 + rotation_time;
    long double recall = 1.0f * correct / total;
    
    // cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query << " " << adsampling::tot_dimension + adsampling::tot_full_dist * vecdim << endl;
    cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query << " ||| comp. ratio: " << (long double)adsampling::tot_full_dist / (long double)adsampling::tot_dist_calculation 
    << " ||| preprocess time: " << rotation_time  << " ||| # total dist: " << adsampling::tot_dist_calculation << " ||| total dimensions: "<< adsampling::tot_dimension 
    << endl;
    return ;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    vector<size_t> efs{500, 800, 1500};
        // ProfilerStart("../prof/svd-profile.prof");
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive);
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
    int iarg = -1;
    opterr = 1;    //getopt error message (off: 0)

    // 0: original HNSW, 1: HNSW++ 2: HNSW+ 3: PAA 4: LSH 5: SVD 6: PQ 7: OPQ 8: Finger
    //                             20:HNSW+-keep
    //                                                                 71: OPQ++
    int randomize = 4;
    string exp_name = "LSH16";
    // string exp_name = "ADSKeep2.1-32";
    // string exp_name = "OPQ+8-256";
    // string exp_name = "SIMD";
    // string exp_name = "";
    int subk=20;
    float ads_epsilon0 = 2.1;
    float ads_delta_d = 32;
    int paa_segment = 96;
    int lsh_dim = 16;
    double lsh_p_tau = 0.9;
    int pq_m = 8;
    int pq_ks = 256;
    float pq_epsilon = 0.9;

    string base_path_str = "../data";
    string result_base_path_str = "../results";
    string data_str = "gist";   // dataset name
    string ef_str = "500";
    string M_str ="16";
    string index_path_str = base_path_str + "/" + data_str + "/" + data_str + "_ef" + ef_str + "_M" + M_str + ".index";
    string query_path_str = base_path_str + "/" + data_str + "/" + data_str + "_query.fvecs";
    string result_path_str = result_base_path_str + "/" + data_str + "/" + data_str + "_ef" + ef_str + "_M" + M_str + "_" + exp_name + ".log";
    string groundtruth_path_str = base_path_str + "/" + data_str + "/" + data_str + "_groundtruth.ivecs";
    string trans_path_str = base_path_str + "/" + data_str + "/O.fvecs";
    string transed_data_path_str = base_path_str + "/" + data_str + "/O" + data_str + "_base.fvecs";
    string lsh_trans_path_str = base_path_str + "/" + data_str + "/LSH" + to_string(lsh_dim) + ".fvecs";
    string lsh_path_str = base_path_str + "/" + data_str + "/LSH" + to_string(lsh_dim) + "_" + data_str + "_base.fvecs";
    string svd_trans_path_str = base_path_str + "/" + data_str + "/SVD.fvecs";
    string svd_path_str = base_path_str + "/" + data_str + "/SVD_" + data_str + "_base.fvecs";
    string paa_path_str = base_path_str + "/" + data_str + "/PAA_" + to_string(paa_segment) + "_" + data_str + "_base.fvecs";
    string pq_codebook_path_str = base_path_str + "/" + data_str + "/PQ_codebook_" + to_string(pq_m) + "_" + to_string(pq_ks) + ".fdat";
    string pq_codes_path_str = base_path_str + "/" + data_str + "/PQ_" + to_string(pq_m) + "_" + to_string(pq_ks) + "_" + data_str + "_base.ivecs";
    string opq_rotation_path_str = base_path_str + "/" + data_str + "/OPQ_rotation_" + to_string(pq_m) + "_" + to_string(pq_ks)  + ".fvecs";
    string opq_codebook_path_str = base_path_str + "/" + data_str + "/OPQ_codebook_" + to_string(pq_m) + "_" + to_string(pq_ks) + ".fdat";
    string opq_codes_path_str = base_path_str + "/" + data_str + "/OPQ_" + to_string(pq_m) + "_" + to_string(pq_ks) + "_" + data_str + "_base.ivecs";
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
    char svd_path[256] = "";
    strcpy(svd_path, svd_path_str.c_str());
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


    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg)randomize = atoi(optarg);
                break;
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

    // InnerProductSpace ip_space(Q.d);

    cout << "result path: "<< result_path << endl;

    // adsampling::D = Q.d;
    freopen(result_path,"a",stdout);
    if(randomize == 1 || randomize == 2){
        Matrix<float> P(transformation_path);
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
    }else if(randomize == 20){
        Matrix<float> P(transformation_path);
        adsampling::project_table = Matrix<float>(transed_data_path);
        StopW stopw = StopW();
        adsampling::queries_project = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
    }else if(randomize == 3){
        paa::paa_table = Matrix<float>(paa_path);
        StopW stopw = StopW();
        paa::queries_paa = to_paa(Q, paa_segment);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        paa::D = Q.d;
        paa::segment_num = paa_segment;
        paa::len_per_seg = paa::D / paa::segment_num;
    }else if(randomize == 4){
        Matrix<float> P(lsh_trans_path);
        lsh::lsh_table = Matrix<float>(lsh_path);
        StopW stopw = StopW();
        lsh::queries_lsh = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        lsh::D = Q.d;
        lsh::probQ = lsh_p_tau;
        lsh::lowdim = P.d;
        lsh::initialize();
    }else if(randomize == 5){
        Matrix<float> P(svd_trans_path);
        // svd::svd_table = Matrix<float>(svd_path);
        StopW stopw = StopW();
        svd::queries_svd = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        svd::D = Q.d;
    }else if(randomize == 6 || randomize == 61){
        pq::M = pq_m;
        pq::Ks = pq_ks;
        pq::sub_vec_len = Q.d / pq_m;
        Matrix<float> tmp(pq_codebook_path, true);
        pq::init_codebook(tmp);
        pq::pq_codes_base = Matrix<int>(pq_codes_path);
        pq::pq_codes_base.M = pq::pq_codes_base.d;
        // svd::svd_table = Matrix<float>(svd_path);
        StopW stopw = StopW();
        pq::calc_dist_book(Q);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        pq::D = Q.d;
        pq::epsilon = pq_epsilon;
    }else if(randomize == 7 || randomize == 71){
        pq::M = pq_m;
        pq::Ks = pq_ks;
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
        pq::D = Q.d;
        pq::epsilon = pq_epsilon;
    }
    
    L2Space l2space(Q.d);
    
    HierarchicalNSW<float> *appr_alg = new HierarchicalNSW<float>(&l2space, index_path, false);
    size_t k = G.d;

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;

    get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
    // ProfilerStart("../prof/svd-profile.prof");
    test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, randomize);
    // ProfilerStop();

    return 0;
}
