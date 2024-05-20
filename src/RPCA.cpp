//
// Created by xuhui on 24-5-9.
//
#include "../include/RPCA.h"

RPCA::RPCA() {

}

// 四舍五入
double RPCA::round(double r) {
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}

// 将小于0的元素全部设为0
Eigen::MatrixXd RPCA::setNegNum2Zero(Eigen::MatrixXd in_Mat) {
    Eigen::MatrixXd out_Mat = Eigen::MatrixXd::Zero(in_Mat.rows(), in_Mat.cols());
    for (int i = 0; i < in_Mat.rows(); i++) {
        for (int j = 0; j < in_Mat.cols(); j++) {
            if (in_Mat(i, j) >= 0) {
                out_Mat(i, j) = in_Mat(i, j);
            }
        }
    }
    return out_Mat;
}

// 将大于0的元素全部设为0
Eigen::MatrixXd RPCA::setPosNum2Zero(Eigen::MatrixXd in_Mat) {
    Eigen::MatrixXd out_Mat = Eigen::MatrixXd::Zero(in_Mat.rows(), in_Mat.cols());
    for (int i = 0; i < in_Mat.rows(); i++) {
        for (int j = 0; j < in_Mat.cols(); j++) {
            if (in_Mat(i, j) < 0) {
                out_Mat(i, j) = in_Mat(i, j);
            }
        }
    }
    return out_Mat;
}

// 根据不同情况选择SVD方法
int RPCA::choosvd(int n, int d) {
    int y;
    if (n <= 100) {
        if (d / n <= 0.02) {
            y = 1;
        } else {
            y = 0;
        }
    } else if (n <= 200) {
        if (d / n <= 0.06) {
            y = 1;
        } else {
            y = 0;
        }
    } else if (n <= 300) {
        if (d / n <= 0.26) {
            y = 1;
        } else {
            y = 0;
        }
    } else if (n <= 400) {
        if (d / n <= 0.28) {
            y = 1;
        } else {
            y = 0;
        }
    } else if (n <= 500) {
        if (d / n <= 0.34) {
            y = 1;
        } else {
            y = 0;
        }
    } else {
        if (d / n <= 0.38) {
            y = 1;
        } else {
            y = 0;
        }
    }
    return y;
}

// 统计矩阵大于等于阈值的元素个数
int RPCA::numLargerThanTh(Eigen::MatrixXd in_mat, double th) {
    int index = 0;
    for (int i = 0; i < in_mat.rows(); i++) {
        for (int j = 0; j < in_mat.cols(); j++) {
            if (in_mat(i, j) >= th) {
                index++;
            }
        }
    }
    return index;
}

// Matlab索引与Eigen索引转换
void RPCA::cvtIndex(int start, int end, int &cvt_start, int &cvt_end) {
    cvt_start = start - 1;
    if (cvt_start < 0) {
        cvt_start = 0;
    }

    cvt_end = end - start + 1;
    if (cvt_end == 0) {
        cvt_end = 1;
    }
}

// Eigen的BDC SVD分解方法
void RPCA::bdc_svd(Eigen::MatrixXd in_Mat, Eigen::MatrixXd &U, Eigen::MatrixXd &S, Eigen::MatrixXd &V) {
    // 构造SVD对象，除了BDC还有Jacobian等方法
    Eigen::BDCSVD<Eigen::MatrixXd> svd1(in_Mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Eigen得到的是奇异值向量以及V，所以需要构造对角阵并对V转置
    U = svd1.matrixU();
    S = svd1.singularValues().asDiagonal();
    V = svd1.matrixV();
}

// Truncated SVD分解方法
void
RPCA::truncated_svd(Eigen::MatrixXd in_mat, Eigen::MatrixXd &U_mat, Eigen::MatrixXd &S_mat, Eigen::MatrixXd &V_mat) {
    // Truncated SVD输入的是数组，因此转换一下
    double *in_mat_ptr = in_mat.data();
    int m = in_mat.rows();
    int n = in_mat.cols();
    double *un = new double[m * n];
    double *sn = new double[n * n];
    double *v = new double[n * n];

    // m > n, row > col的情况
    svd_truncated_u(m, n, in_mat_ptr, un, sn, v);

    // 重新构造Eigen矩阵
    U_mat = Eigen::Map<Eigen::MatrixXd>(un, m, n);
    S_mat = Eigen::Map<Eigen::MatrixXd>(sn, n, n);
    V_mat = Eigen::Map<Eigen::MatrixXd>(v, n, n);
}

// Armadillo SVD分解方法
void
RPCA::armadillo_svd(Eigen::MatrixXd in_mat, Eigen::MatrixXd &U_mat, Eigen::MatrixXd &S_mat, Eigen::MatrixXd &V_mat) {
    // Eigen转Armadillo
    arma::mat in_mat_arma = arma::mat(in_mat.data(), in_mat.rows(), in_mat.cols(),
                                      false, false);

    // svd
    arma::mat U;
    arma::vec s;
    arma::mat V;
    svd_econ(U, s, V, in_mat_arma);

    // Armadillo转Eigen
    U_mat = Eigen::Map<Eigen::MatrixXd>(U.memptr(),
                                        U.n_rows,
                                        U.n_cols);
    S_mat = Eigen::Map<Eigen::MatrixXd>(s.memptr(),
                                        s.n_rows,
                                        s.n_cols);
    V_mat = Eigen::Map<Eigen::MatrixXd>(V.memptr(),
                                        V.n_rows,
                                        V.n_cols);
}

// BDC SVD RPCA
void RPCA::inexact_alm_rpca_bdc_svd(Eigen::MatrixXd D,
                                    double in_lambda, double in_tol, int in_maxIter,
                                    Eigen::MatrixXd &A_hat, Eigen::MatrixXd &E_hat, int &iter) {
    int m = D.rows();   // m是行数
    int n = D.cols();   // n是列数

    // 控制循环的相关变量，如果传入的值是-1,则自动算出，否则就是传入的值
    // lambda默认值为1/sqrt(m)
    // tol默认值为1e-7
    // maxIter默认值为100
    double lambda, tol, maxIter;

    if (in_lambda == -1) {
        lambda = 1 / sqrt(m * 1.0);
    } else {
        lambda = in_lambda;
    }

    if (in_tol == -1) {
        tol = 1e-7;
    } else {
        tol = in_tol;
    }

    if (in_maxIter == -1) {
        maxIter = 100;
    } else {
        maxIter = in_maxIter;
    }

    Eigen::MatrixXd Y;
    Y = D;
    double norm_two, norm_inf, dual_norm;
    Eigen::BDCSVD<Eigen::MatrixXd> svdY(Y, Eigen::ComputeThinU | Eigen::ComputeThinV);
    norm_two = svdY.singularValues()[0];
    norm_inf = Y.cwiseAbs().maxCoeff() / lambda;

    dual_norm = max(norm_two, norm_inf);
    Y = Y / dual_norm;

    A_hat = Eigen::MatrixXd::Zero(m, n);
    E_hat = Eigen::MatrixXd::Zero(m, n);
    double mu = 1.25 / norm_two;
    double mu_bar = mu * 1e7;
    double rho = 1.5;
    double d_norm = D.lpNorm<2>();

    iter = 0;
    int total_svd = 0;
    bool converged = false;
    double stopCriterion = 1;
    double sv = 10;

    // 开始迭代
    while (!converged) {
        iter += 1;

        Eigen::MatrixXd temp_T = D - A_hat + (1 / mu) * Y;
        Eigen::MatrixXd coeff = Eigen::MatrixXd::Constant(temp_T.rows(), temp_T.cols(), lambda / mu);
        E_hat = setNegNum2Zero(temp_T - coeff);
        E_hat = E_hat + setPosNum2Zero(temp_T + coeff);

        Eigen::MatrixXd U, S, V;
        if (choosvd(n, sv) == 1) {
            bdc_svd(D - E_hat + (1 / mu) * Y, U, S, V);
        } else {
            bdc_svd(D - E_hat + (1 / mu) * Y, U, S, V);
        }

        int svp = numLargerThanTh(S, 1 / mu);

        if (svp < sv) {
            sv = min(svp + 1, n);
        } else {
            sv = min(svp + int(round(0.05 * n)), n);
        }

        // 核心问题还是Matlab和Eigen API的索引方式不同，需要转换
        int start, end;
        cvtIndex(1, svp, start, end);
        Eigen::MatrixXd U_mat = U.block(0, start, U.rows(), end);
        Eigen::MatrixXd V_mat = V.block(0, start, V.rows(), end);

        int start2, end2;
        cvtIndex(1, svp, start2, end2);
        Eigen::VectorXd diag = S.diagonal().segment(start2, end2);
        Eigen::VectorXd coeff2 = Eigen::VectorXd::Constant(diag.rows(), 1 / mu);
        Eigen::MatrixXd diag_mat = (diag - coeff2).asDiagonal();

        A_hat = U_mat * diag_mat * V_mat.transpose();

        total_svd++;

        Eigen::MatrixXd Z = D - A_hat - E_hat;

        Y = Y + mu * Z;
        mu = min(mu * rho, mu_bar);

        stopCriterion = Z.lpNorm<2>() / d_norm;
        if (stopCriterion < tol) {
            converged = true;
        }

        if (total_svd % 10 == 0) {
            Eigen::FullPivLU<Eigen::MatrixXd> A_lu(A_hat);
            double ra = A_lu.rank();
            double e0 = E_hat.cwiseAbs().count();

            cout << "#svd " << total_svd <<
                 " r(A) " << ra <<
                 " |E|_0 " << e0 <<
                 " stopCriterion " << stopCriterion << endl;
        }
        Eigen::FullPivLU<Eigen::MatrixXd> A_lu(A_hat);
        A_lu.rank();

        if (!converged && iter >= maxIter) {
            cout << "Maximum iterations reached" << endl;
            converged = true;
        }
    }
}

// Truncated SVD RPCA
void RPCA::inexact_alm_rpca_truncated_svd(Eigen::MatrixXd D,
                                          double in_lambda, double in_tol, int in_maxIter,
                                          Eigen::MatrixXd &A_hat, Eigen::MatrixXd &E_hat, int &iter) {
    int m = D.rows();   // m是行数
    int n = D.cols();   // n是列数

    // 控制循环的相关变量，如果传入的值是-1,则自动算出，否则就是传入的值
    // lambda默认值为1/sqrt(m)
    // tol默认值为1e-7
    // maxIter默认值为100
    double lambda, tol, maxIter;

    if (in_lambda == -1) {
        lambda = 1 / sqrt(m * 1.0);
    } else {
        lambda = in_lambda;
    }

    if (in_tol == -1) {
        tol = 1e-7;
    } else {
        tol = in_tol;
    }

    if (in_maxIter == -1) {
        maxIter = 100;
    } else {
        maxIter = in_maxIter;
    }

    Eigen::MatrixXd Y, tmp_U, tmp_S, tmp_V;
    Y = D;
    double norm_two, norm_inf, dual_norm;
    truncated_svd(Y, tmp_U, tmp_S, tmp_V);
    norm_two = tmp_S(0, 0);
    norm_inf = Y.cwiseAbs().maxCoeff() / lambda;

    dual_norm = max(norm_two, norm_inf);
    Y = Y / dual_norm;

    A_hat = Eigen::MatrixXd::Zero(m, n);
    E_hat = Eigen::MatrixXd::Zero(m, n);
    double mu = 1.25 / norm_two;
    double mu_bar = mu * 1e7;
    double rho = 1.5;
    double d_norm = D.lpNorm<2>();

    iter = 0;
    int total_svd = 0;
    bool converged = false;
    double stopCriterion = 1;
    double sv = 10;

    // 开始迭代
    while (!converged) {
        iter += 1;
        //cout << "iter " << iter << endl;

        Eigen::MatrixXd temp_T = D - A_hat + (1 / mu) * Y;
        Eigen::MatrixXd coeff = Eigen::MatrixXd::Constant(temp_T.rows(), temp_T.cols(), lambda / mu);
        E_hat = setNegNum2Zero(temp_T - coeff);
        E_hat = E_hat + setPosNum2Zero(temp_T + coeff);

        Eigen::MatrixXd U, S, V;
        if (choosvd(n, sv) == 1) {
            truncated_svd(D - E_hat + (1 / mu) * Y, U, S, V);
        } else {
            truncated_svd(D - E_hat + (1 / mu) * Y, U, S, V);
        }

        int svp = numLargerThanTh(S, 1 / mu);

        if (svp < sv) {
            sv = min(svp + 1, n);
        } else {
            sv = min(svp + int(round(0.05 * n)), n);
        }

        // 核心问题还是Matlab和Eigen API的索引方式不同，需要转换
        int start, end;
        cvtIndex(1, svp, start, end);
        Eigen::MatrixXd U_mat = U.block(0, start, U.rows(), end);
        Eigen::MatrixXd V_mat = V.block(0, start, V.rows(), end);

        int start2, end2;
        cvtIndex(1, svp, start2, end2);
        Eigen::VectorXd diag = S.diagonal().segment(start2, end2);
        Eigen::VectorXd coeff2 = Eigen::VectorXd::Constant(diag.rows(), 1 / mu);
        Eigen::MatrixXd diag_mat = (diag - coeff2).asDiagonal();

        A_hat = U_mat * diag_mat * V_mat.transpose();

        total_svd++;

        Eigen::MatrixXd Z = D - A_hat - E_hat;

        Y = Y + mu * Z;
        mu = min(mu * rho, mu_bar);

        stopCriterion = Z.lpNorm<2>() / d_norm;
        if (stopCriterion < tol) {
            converged = true;
        }

        if (total_svd % 5 == 0) {
            Eigen::FullPivLU<Eigen::MatrixXd> A_lu(A_hat);
            double ra = A_lu.rank();
            double e0 = E_hat.cwiseAbs().count();

            cout << "#svd " << total_svd <<
                 " r(A) " << ra <<
                 " |E|_0 " << e0 <<
                 " stopCriterion " << stopCriterion << endl;
        }
        Eigen::FullPivLU<Eigen::MatrixXd> A_lu(A_hat);
        A_lu.rank();

        if (!converged && iter >= maxIter) {
            cout << "Maximum iterations reached" << endl;
            converged = true;
        }
    }
}

// Armadillo SVD RPCA
void RPCA::inexact_alm_rpca_armadillo_svd(Eigen::MatrixXd D,
                                          double in_lambda, double in_tol, int in_maxIter,
                                          Eigen::MatrixXd &A_hat, Eigen::MatrixXd &E_hat, int &iter) {
    int m = D.rows();   // m是行数
    int n = D.cols();   // n是列数

    // 控制循环的相关变量，如果传入的值是-1,则自动算出，否则就是传入的值
    // lambda默认值为1/sqrt(m)
    // tol默认值为1e-7
    // maxIter默认值为100
    double lambda, tol, maxIter;

    if (in_lambda == -1) {
        lambda = 1 / sqrt(m * 1.0);
    } else {
        lambda = in_lambda;
    }

    if (in_tol == -1) {
        tol = 1e-7;
    } else {
        tol = in_tol;
    }

    if (in_maxIter == -1) {
        maxIter = 100;
    } else {
        maxIter = in_maxIter;
    }

    Eigen::MatrixXd Y, tmp_U, tmp_S, tmp_V;
    Y = D;
    double norm_two, norm_inf, dual_norm;
    armadillo_svd(Y, tmp_U, tmp_S, tmp_V);
    norm_two = tmp_S(0, 0);
    norm_inf = Y.cwiseAbs().maxCoeff() / lambda;

    dual_norm = max(norm_two, norm_inf);
    Y = Y / dual_norm;

    A_hat = Eigen::MatrixXd::Zero(m, n);
    E_hat = Eigen::MatrixXd::Zero(m, n);
    double mu = 1.25 / norm_two;
    double mu_bar = mu * 1e7;
    double rho = 1.5;
    double d_norm = D.lpNorm<2>();

    iter = 0;
    int total_svd = 0;
    bool converged = false;
    double stopCriterion = 1;
    double sv = 10;

    // 开始迭代
    while (!converged) {
        iter += 1;
        //cout << "iter " << iter << endl;

        Eigen::MatrixXd temp_T = D - A_hat + (1 / mu) * Y;
        Eigen::MatrixXd coeff = Eigen::MatrixXd::Constant(temp_T.rows(), temp_T.cols(), lambda / mu);
        E_hat = setNegNum2Zero(temp_T - coeff);
        E_hat = E_hat + setPosNum2Zero(temp_T + coeff);

        Eigen::MatrixXd U, S, V;
        if (choosvd(n, sv) == 1) {
            armadillo_svd(D - E_hat + (1 / mu) * Y, U, S, V);
        } else {
            armadillo_svd(D - E_hat + (1 / mu) * Y, U, S, V);
        }

        int svp = numLargerThanTh(S, 1 / mu);

        if (svp < sv) {
            sv = min(svp + 1, n);
        } else {
            sv = min(svp + int(round(0.05 * n)), n);
        }

        // 核心问题还是Matlab和Eigen API的索引方式不同，需要转换
        int start, end;
        cvtIndex(1, svp, start, end);
        Eigen::MatrixXd U_mat = U.block(0, start, U.rows(), end);
        Eigen::MatrixXd V_mat = V.block(0, start, V.rows(), end);

        int start2, end2;
        cvtIndex(1, svp, start2, end2);
        Eigen::VectorXd diag = S.diagonal().segment(start2, end2);
        Eigen::VectorXd coeff2 = Eigen::VectorXd::Constant(diag.rows(), 1 / mu);
        Eigen::MatrixXd diag_mat = (diag - coeff2).asDiagonal();

        A_hat = U_mat * diag_mat * V_mat.transpose();

        total_svd++;

        Eigen::MatrixXd Z = D - A_hat - E_hat;

        Y = Y + mu * Z;
        mu = min(mu * rho, mu_bar);

        stopCriterion = Z.lpNorm<2>() / d_norm;
        if (stopCriterion < tol) {
            converged = true;
        }

        if (total_svd % 5 == 0) {
            Eigen::FullPivLU<Eigen::MatrixXd> A_lu(A_hat);
            double ra = A_lu.rank();
            double e0 = E_hat.cwiseAbs().count();

            cout << "#svd " << total_svd <<
                 " r(A) " << ra <<
                 " |E|_0 " << e0 <<
                 " stopCriterion " << stopCriterion << endl;
        }
        Eigen::FullPivLU<Eigen::MatrixXd> A_lu(A_hat);
        A_lu.rank();

        if (!converged && iter >= maxIter) {
            cout << "Maximum iterations reached" << endl;
            converged = true;
        }
    }
}

// 将影像按列叠加成一个一维向量
cv::Mat RPCA::stackCols(cv::Mat in_img) {
    int rows = in_img.rows;
    int cols = in_img.cols;
    cv::Mat stacked_vec = in_img.col(0);
    for (int i = 1; i < cols; i++) {
        vconcat(stacked_vec, in_img.col(i), stacked_vec);
    }
    return stacked_vec;
}

// 将多个向量构造成大矩阵
cv::Mat RPCA::joinImgVectors(vector<cv::Mat> vectors) {
    cv::Mat final_mat = vectors[0];
    for (int i = 1; i < vectors.size(); i++) {
        hconcat(final_mat, vectors[i], final_mat);
    }
    return final_mat;
}

// 由一维向量恢复二维影像
cv::Mat RPCA::restoreImg(cv::Mat in_col, int img_w, int img_h) {
    cv::Mat img;
    img = in_col(cv::Range(0, img_h), cv::Range::all());
    for (int i = 1; i < img_w; i++) {
        cv::Mat tmp_col = in_col(cv::Range(i * img_h, (i + 1) * img_h), cv::Range::all());
        hconcat(img, tmp_col, img);
    }
    return img;
}

// 根据高维矩阵恢复所有二维影像
vector<cv::Mat> RPCA::parseMat2Img(cv::Mat mat, int img_w, int img_h) {
    vector<cv::Mat> img_vec;
    int rows = mat.rows;
    int cols = mat.cols;
    for (int i = 0; i < cols; i++) {
        cv::Mat restore_img = restoreImg(mat.col(i), img_w, img_h);
        img_vec.push_back(restore_img);
    }
    return img_vec;
}

// 将Mat转换为单通道IplImage
IplImage *mat2IplImg1C(cv::Mat img) {
    int w = img.cols;
    int h = img.rows;
    IplImage *dst = cvCreateImage(cvSize(w, h), 8, 1);
    for (int j = 0; j < w; ++j) {
        for (int i = 0; i < h; ++i) {
            int p = img.at<uchar>(i, j);
            CV_IMAGE_ELEM(dst, uchar, i, j * 1 + 0) = p;
        }
    }
    return dst;
}

//遍历指定路径下的所有文件，将文件的路径存于vector中
void RPCA::GetFileNames(string path, vector<string> &paths, vector<string> &names, vector<string> &files) {
    DIR *pDir;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str()))) {
        cout << "error：Folder doesn't Exist!" << endl;
        return;
    }
    while ((ptr = readdir(pDir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            paths.push_back(path);
            names.push_back(ptr->d_name);
            files.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
}

// 筛选指定类型的文件
void RPCA::findAllFiles(string root_dir, string filter,
                        vector<string> &paths, vector<string> &names, vector<string> &files) {
    vector<string> all_paths, all_names, all_files;
    GetFileNames(root_dir, all_paths, all_names, all_files);
    string::size_type idx;

    for (int i = 0; i < all_names.size(); i++) {
        idx = all_names[i].find(filter);
        if (idx != string::npos) {
            paths.push_back(all_paths[i]);
            names.push_back(all_names[i]);
            files.push_back(all_files[i]);
        }
    }

    sort(paths.begin(), paths.end());
    sort(names.begin(), names.end());
    sort(files.begin(), files.end());
}

vector<vector<cv::Mat>> RPCA::run(vector<string> img_files, int flag, double scale_factor,
                                  double lambda, double toler, int maxIter) {
    // 构造矩阵(将输入影像变成一维向量)
    vector<cv::Mat> ori_imgs, col_imgs;
    int img_width;
    int img_height;

    for (int i = 0; i < img_files.size(); ++i) {
        cv::Mat img = cv::imread(img_files[i], cv::IMREAD_GRAYSCALE);
        cv::Mat img_resize;

        img_width = img.cols;
        img_height = img.rows;
        // 如果不是原始尺寸就进行缩放
        if (scale_factor != 1) {
            resize(img, img_resize, cv::Size(int(scale_factor * img_width), int(scale_factor * img_height)));
            ori_imgs.push_back(img_resize);
            col_imgs.push_back(stackCols(img_resize));
        } else {
            ori_imgs.push_back(img);
            col_imgs.push_back(stackCols(img));
        }
    }

    // 构造矩阵(将多个一维向量合并，构成超高维矩阵)
    cv::Mat final_mat = joinImgVectors(col_imgs);

    // 矩阵数据类型转换(OpenCV Mat(uint8) -> OpenCV Mat(double) -> Eigen Matrix(double))
    cv::Mat mat_double(final_mat.rows, final_mat.cols, CV_64F);
    final_mat.convertTo(mat_double, CV_64F);
    Eigen::MatrixXd mat_eigen;
    cv2eigen(mat_double, mat_eigen);

    // 计算稀疏部分权重
    if (lambda == -1) {
        lambda = 1 / sqrt(img_width * img_height);
    }

    // RPCA
    Eigen::MatrixXd A, E;
    int iter;

    auto start_time = std::chrono::high_resolution_clock::now();
    if (flag == 0) {
        // armadillo
        inexact_alm_rpca_armadillo_svd(mat_eigen, lambda, toler, maxIter, A, E, iter);
    } else if (flag == 1) {
        // truncated
        inexact_alm_rpca_truncated_svd(mat_eigen, lambda, toler, maxIter, A, E, iter);
    } else if (flag == 2) {
        // eigen
        inexact_alm_rpca_bdc_svd(mat_eigen, lambda, toler, maxIter, A, E, iter);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // 转换结果(Eigen Matrix(double) -> OpenCV Mat(double))
    cv::Mat A_mat, E_mat;
    eigen2cv(A, A_mat);
    eigen2cv(E, E_mat);

    // 解析结果(一维向量到二维矩阵)
    vector<cv::Mat> A_mats = parseMat2Img(A_mat, ori_imgs[0].cols, ori_imgs[0].rows);
    vector<cv::Mat> E_mats = parseMat2Img(E_mat, ori_imgs[0].cols, ori_imgs[0].rows);

    // 转换为8bit影像以方便后续处理
    vector<cv::Mat> Low_rank_8bit, Sparse_8bit;
    for (int j = 0; j < A_mats.size(); ++j) {
        cv::Mat tmp_low, tmp_sparse;

        // 寻找最大最小值
        double max_v_low, min_v_low, max_v_sparse, min_v_sparse;
        minMaxLoc(A_mats[j], &min_v_low, &max_v_low);
        minMaxLoc(E_mats[j], &min_v_sparse, &max_v_sparse);

        // 在这一步转换的时候损失了数据
        // 因为Sparse部分是有负值的，而直接这样转换所有小于0的数据都被设为0了
        A_mats[j].convertTo(tmp_low, CV_8U);
        // 所以，可以把所有小于0的值变为0，使得最小值为0，这样就可以正常计算了
        E_mats[j] = E_mats[j] + abs(min_v_sparse);
        E_mats[j].convertTo(tmp_sparse, CV_8U);

        // 添加解析好的影像到vector中
        Low_rank_8bit.push_back(tmp_low);
        Sparse_8bit.push_back(tmp_sparse);
    }

    vector<vector<cv::Mat>> results;
    results.push_back(ori_imgs);
    results.push_back(Low_rank_8bit);
    results.push_back(Sparse_8bit);

    return results;
}