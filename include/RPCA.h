//
// Created by xuhui on 24-5-9.
//

// 基础功能相关引用
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <string>
#include <algorithm>
#include <dirent.h>
#include <cstring>
#include <chrono>

// Eigen相关引用
#include <Eigen/Core>
#include <Eigen/Dense>

// Armadillo相关引用
#include <armadillo>

// OpenCV相关引用
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>

// Truncated SVD
#include "svd_truncated.h"


using namespace std;

#ifndef CODE_RPCA_H
#define CODE_RPCA_H

class RPCA {

public:
    // constructor
    RPCA();

    // tool functions for element-wise operations
    double round(double r);
    Eigen::MatrixXd setNegNum2Zero(Eigen::MatrixXd in_Mat);
    Eigen::MatrixXd setPosNum2Zero(Eigen::MatrixXd in_Mat);
    int choosvd(int n, int d);
    int numLargerThanTh(Eigen::MatrixXd in_mat, double th);
    void cvtIndex(int start, int end, int &cvt_start, int &cvt_end);

    // svd functions
    void bdc_svd(Eigen::MatrixXd in_Mat, Eigen::MatrixXd &U, Eigen::MatrixXd &S, Eigen::MatrixXd &V);
    void truncated_svd(Eigen::MatrixXd in_mat, Eigen::MatrixXd &U_mat, Eigen::MatrixXd &S_mat, Eigen::MatrixXd &V_mat);
    void armadillo_svd(Eigen::MatrixXd in_mat, Eigen::MatrixXd &U_mat, Eigen::MatrixXd &S_mat, Eigen::MatrixXd &V_mat);

    // RPCA functions
    void inexact_alm_rpca_bdc_svd(Eigen::MatrixXd D,
                                  double in_lambda, double in_tol, int in_maxIter,
                                  Eigen::MatrixXd &A_hat, Eigen::MatrixXd &E_hat, int &iter);
    void inexact_alm_rpca_truncated_svd(Eigen::MatrixXd D,
                                        double in_lambda, double in_tol, int in_maxIter,
                                        Eigen::MatrixXd &A_hat, Eigen::MatrixXd &E_hat, int &iter);
    void inexact_alm_rpca_armadillo_svd(Eigen::MatrixXd D,
                                        double in_lambda, double in_tol, int in_maxIter,
                                        Eigen::MatrixXd &A_hat, Eigen::MatrixXd &E_hat, int &iter);

    // tool functions for input and output data
    cv::Mat stackCols(cv::Mat in_img);
    cv::Mat joinImgVectors(vector<cv::Mat> vectors);
    cv::Mat restoreImg(cv::Mat in_col, int img_w, int img_h);
    vector<cv::Mat> parseMat2Img(cv::Mat mat, int img_w, int img_h);

    // file-related functions
    void GetFileNames(string path, vector<string> &paths, vector<string> &names, vector<string> &files);
    void findAllFiles(string root_dir, string filter,
                      vector<string> &paths, vector<string> &names, vector<string> &files);

    // wrapper for use
    vector<vector<cv::Mat>> run(vector<string> img_files,
                                int flag = 1,
                                double scale_factor = 1.0,
                                double lambda = -1,
                                double toler = -1,
                                int maxIter = -1);
};


#endif //CODE_RPCA_H
