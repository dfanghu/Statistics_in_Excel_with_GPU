#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "Windows.h"
#include "XLCALL.H"
#include "FRAMEWRK.H"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "magma.h"
#include "magma_lapack.h"
#include "magma_timer.h"
using namespace std;
extern ofstream logfile;
extern XCHAR* errmsg;
extern HANDLE hArray;
extern int errortype;
extern cudaError_t cudaErrorCode;
extern cublasStatus_t cublasErrorCode;
extern cusolverStatus_t cusolverErrorCode;
#define max(a,b) (((a)<(b))?(b):(a))
#define SGN(u) (((u) > 0) ? 1 : (((u) < 0) ? -1 : 0))
#define asrt(lbl,Lmsg,x) if(!(x)){errmsg = Lmsg; goto lbl;}
#define cuAsrt(lbl,x) if(cudaSuccess != (cudaErrorCode = (x)) && (errortype = 1)) goto lbl
#define cublasAsrt(lbl,x) if(CUBLAS_STATUS_SUCCESS != (cublasErrorCode = (x))  && (errortype = 2)) goto lbl
#define cusolverAsrt(lbl,x) if(CUSOLVER_STATUS_SUCCESS != (cusolverErrorCode = (x))  && (errortype = 3)) goto lbl

XCHAR* pc2pwc(XCHAR* pwc, const char* pc);
string pwc2str(const XCHAR* pwc);
XCHAR* str2pwc(XCHAR* pwc, std::string s);
void* pc2hex(void* p, const char* s);
void* pwc2hex(void* p, const XCHAR* ws);
string hex2str(void* p);
char* hex2pc(char* pc, void* p);
XCHAR* hex2pwc(XCHAR* pwc, void* p);
XCHAR* hex2pwc(XCHAR* pwc, void* p);

void alloc_cpu(double** a, double** da, magma_int_t m);
void free_cpu(double** a, double** da);
void alloc_pinned(double** a, double** da, magma_int_t m);
void free_pinned(double**a, double** da);

LPXLOPER12 WINAPI gpu_cudaDeviceReset();
LPXLOPER12 WINAPI gpu_cudaGetLastError();
LPXLOPER12 WINAPI gpu_cudaGetLastErrorString();
LPXLOPER12 WINAPI gpu_cudaSetDevice(int dev);
LPXLOPER12 WINAPI gpu_cudaGetDeviceCount();
LPXLOPER12 WINAPI gpu_totalGlobalMemInMB();
LPXLOPER12 WINAPI gpu_cudaVersion();
LPXLOPER12 WINAPI gpu_name();
LPXLOPER12 WINAPI gpu_cudaMalloc(FP12 x);
LPXLOPER12 WINAPI gpu_cudaFree(XCHAR* str_d_x);
LPXLOPER12 WINAPI gpu_getGpuData(XCHAR* str_d_x, int m, int n);

const char* cublasGetErrorString(cublasStatus_t status);
string get_errmsg_str();
bool inline default_missing_n_inc(int& n, int& inc, const int leng);

LPXLOPER12 WINAPI gpu_cublasGetLastHandle();
LPXLOPER12 WINAPI gpu_cublasCreate();
LPXLOPER12 WINAPI gpu_cublasDestroy();
LPXLOPER12 WINAPI gpu_cublasIdamax(FP12& x, int incx, int n);
LPXLOPER12 WINAPI gpu_cublasDgemm(double alpha, FP12& A, int transa, FP12& B, int transb, double beta, FP12& C, int m, int k, int n, int lda, int ldb, int ldc);
LPXLOPER12 WINAPI gpu_cublasIdamin(FP12& x, int incx, int n);
LPXLOPER12 WINAPI gpu_cublasDasum(FP12& x, int incx, int n);
LPXLOPER12 WINAPI gpu_cublasDaxpy(double alpha, FP12& x, FP12& y, int incx, int incy, int n);
LPXLOPER12 WINAPI gpu_cublasDscal(FP12& x, double alpha, int incx, int n);
LPXLOPER12 WINAPI gpu_cublasDdot(FP12& x, FP12& y, int incx, int incy, int n);
LPXLOPER12 WINAPI gpu_cublasDnrm2(FP12& x, int incx, int n);
LPXLOPER12 WINAPI gpu_cublasDgemm(double alpha, FP12& A, int transa, FP12& B, int transb, double beta, FP12& C, int m, int k, int n, int lda, int ldb, int ldc);
LPXLOPER12 WINAPI gpu_cublasDsymm(double alpha, FP12& A, int uplo, int side, FP12& B, double beta, FP12& C, int m, int n, int lda, int ldb, int ldc);
LPXLOPER12 WINAPI gpu_cublasDsyrk(double alpha, FP12& A, int trans, double beta, FP12& C, int uplo, int n, int k, int lda, int ldc);
LPXLOPER12 WINAPI gpu_cublasDgeam(double alpha, FP12& A, int transa, double beta, FP12& B, int transb, int m, int n, int lda, int ldb);
LPXLOPER12 WINAPI gpu_cublasDdgmm(FP12& x, FP12& A, int m, int n, int mode, int incx, int lda);
LPXLOPER12 WINAPI gpu_inv(FP12& A);
LPXLOPER12 WINAPI gpu_cusolverDnDsyevd(FP12& A, int uplo, int retEigVec, int eigvalDesc);
LPXLOPER12 WINAPI gpu_eig(FP12& A, int noEigVec, int eigvalAsc);
LPXLOPER12 WINAPI gpu_eigsym(FP12& A);
LPXLOPER12 WINAPI gpu_train_numeric(FP12 D0, FP12 Yobs, FP12 layers, FP12 idTransFnInput, int nIter, int retMode, int useSuppliedInitialW0, LPXLOPER12 W0);