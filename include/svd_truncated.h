//
// Created by xuhui on 24-5-9.
//

# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>

using namespace std;


#ifndef CODE_SVD_TRUNCATED_H
#define CODE_SVD_TRUNCATED_H

int main0();

void daxpy(int n, double da, double dx[], int incx, double dy[], int incy);

double ddot(int n, double dx[], int incx, double dy[], int incy);

double dnrm2(int n, double x[], int incx);

void drot(int n, double x[], int incx, double y[], int incy, double c,
          double s);

void drotg(double *sa, double *sb, double *c, double *s);

void dscal(int n, double sa, double x[], int incx);

int dsvdc(double a[], int lda, int m, int n, double s[], double e[],
          double u[], int ldu, double v[], int ldv, double work[], int job);

void dswap(int n, double x[], int incx, double y[], int incy);

int i4_max(int i1, int i2);

int i4_min(int i1, int i2);

double r8_abs(double x);

double r8_max(double x, double y);

double r8_sign(double x);

void r8mat_print(int m, int n, double a[], string title);

void r8mat_print_some(int m, int n, double a[], int ilo, int jlo, int ihi,
                      int jhi, string title);

double *r8mat_transpose_new(int m, int n, double a[]);

double *r8mat_uniform_01_new(int m, int n, int *seed);

void svd_truncated_u(int m, int n, double a[], double un[], double sn[],
                     double v[]);

void svd_truncated_u_test(int m, int n);

void svd_truncated_v(int m, int n, double a[], double u[], double sm[],
                     double vm[]);

void svd_truncated_v_test(int m, int n);

void timestamp();

#endif //CODE_SVD_TRUNCATED_H
