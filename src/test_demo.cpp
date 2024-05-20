#include <iostream>
#include "../include/RPCA.h"

using namespace std;
using namespace Eigen;
using namespace cv;

int main() {
    // 初始化
    RPCA rpca_obj = *new RPCA();

    // 查找文件
    vector<string> paths, names, files;
    rpca_obj.findAllFiles("../test/imgs", ".jpg", paths, names, files);

    // 基于Armadillo的RPCA
    vector<vector<Mat>> results_arma;
    auto start_time1 = std::chrono::high_resolution_clock::now();
    results_arma = rpca_obj.run(files, 0, 0.7);
    auto end_time1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end_time1 - start_time1;
    std::cout << "arma time: " << elapsed1.count() << "s\n";

    // 基于Truncated SVD的RPCA
    vector<vector<Mat>> results_trun;
    auto start_time2 = std::chrono::high_resolution_clock::now();
    results_trun = rpca_obj.run(files, 1, 0.7);
    auto end_time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end_time2 - start_time2;
    std::cout << "trun time: " << elapsed2.count() << "s\n";

    // 基于Eigen的RPCA
    vector<vector<Mat>> results_eigen;
    auto start_time3 = std::chrono::high_resolution_clock::now();
    results_eigen = rpca_obj.run(files, 2, 0.7);
    auto end_time3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed3 = end_time3 - start_time3;
    std::cout << "eigen time: " << elapsed3.count() << "s\n";


    // 结果输出
    for (int i = 0; i < results_arma[0].size(); ++i) {
        cv::Mat ori_img = results_arma[0][i];
        cv::Mat low_img = results_arma[1][i];
        cv::Mat sparse_img = results_arma[2][i];

        imwrite("../test/results_arma/" + names[i].substr(0, names[i].size() - 4) + "_ori.png", ori_img);
        imwrite("../test/results_arma/" + names[i].substr(0, names[i].size() - 4) + "_low.png", low_img);
        imwrite("../test/results_arma/" + names[i].substr(0, names[i].size() - 4) + "_sparse.png", sparse_img);
    }

    for (int i = 0; i < results_trun[0].size(); ++i) {
        cv::Mat ori_img = results_trun[0][i];
        cv::Mat low_img = results_trun[1][i];
        cv::Mat sparse_img = results_trun[2][i];

        imwrite("../test/results_trun/" + names[i].substr(0, names[i].size() - 4) + "_ori.png", ori_img);
        imwrite("../test/results_trun/" + names[i].substr(0, names[i].size() - 4) + "_low.png", low_img);
        imwrite("../test/results_trun/" + names[i].substr(0, names[i].size() - 4) + "_sparse.png", sparse_img);
    }

    for (int i = 0; i < results_eigen[0].size(); ++i) {
        cv::Mat ori_img = results_eigen[0][i];
        cv::Mat low_img = results_eigen[1][i];
        cv::Mat sparse_img = results_eigen[2][i];

        imwrite("../test/results_eigen/" + names[i].substr(0, names[i].size() - 4) + "_ori.png", ori_img);
        imwrite("../test/results_eigen/" + names[i].substr(0, names[i].size() - 4) + "_low.png", low_img);
        imwrite("../test/results_eigen/" + names[i].substr(0, names[i].size() - 4) + "_sparse.png", sparse_img);
    }

    return 0;
}