#pragma once
#include "gpu_.cuh"

ofstream logfile;
XCHAR* errmsg;
HANDLE hArray;
int errortype;
cudaError_t cudaErrorCode;
cublasStatus_t cublasErrorCode;
cusolverStatus_t cusolverErrorCode;

#define rgFuncsRows 37
#define rgFuncsCols 22
static LPWSTR rgFuncs[rgFuncsRows][rgFuncsCols] = {
  { L"gpu_test", L"QK%", L"gpu_test", L"x", L"1", L"", L"", L"", L"test", L"" },
  { L"gpu_cublasGetLastHandle", L"Q", L"gpu_cublasGetLastHandle", L"", L"1", L"", L"", L"", L"get the last cublas Handle" },
  { L"gpu_cublasCreate", L"Q", L"gpu_cublasCreate", L"", L"1", L"", L"", L"", L"create a cublas Handle" },
  { L"gpu_cublasDestroy", L"Q", L"gpu_cublasDestroy", L"", L"1", L"", L"", L"", L"destroy the last cublas Handle" },
  { L"gpu_xy", L"QK%K%", L"gpu_xy", L"x,y", L"1", L"", L"", L"", L"matrix multiplication", L"Left matrix", L"Right matrix" },
  { L"gpu_xty", L"QK%K%", L"gpu_xty", L"x_before_transpose, y", L"1", L"", L"", L"", L"matrix multiplication", L"Left matrix before transpose", L"Right matrix" },
  { L"gpu_xyt", L"QK%K%", L"gpu_xyt", L"x, y_before_transpose", L"1", L"", L"", L"", L"matrix multiplication", L"Left matrix", L"Right matrix before transpose" },
  { L"gpu_xtyt", L"QK%K%", L"gpu_xtyt", L"x_before_transpose,y_before_transpose", L"1", L"", L"", L"", L"matrix multiplication", L"Left matrix before transpose", L"Right matrix before transpose" },
  { L"gpu_xtx", L"QK%", L"gpu_xtx", L"x", L"1", L"", L"", L"", L"x^T * x", L"matrix" },
  { L"gpu_xxt", L"QK%", L"gpu_xxt", L"x", L"1", L"", L"", L"", L"x * x^T", L"matrix" },
  { L"gpu_cudaDeviceReset", L"Q", L"gpu_cudaDeviceReset", L"", L"1", L"", L"", L"", L"Reset CUDA device" },
  { L"gpu_cudaGetLastError", L"Q", L"gpu_cudaGetLastError", L"", L"1", L"", L"", L"", L"Get last CUDA Error Code" },
  { L"gpu_cudaSetDevice", L"QJ", L"gpu_cudaSetDevice", L"device_id", L"1", L"", L"", L"", L"Set gpu device id to use", L"device id, starting at 0" },
  { L"gpu_cudaGetDeviceCount", L"Q", L"gpu_cudaGetDeviceCount", L"", L"1", L"", L"", L"", L"Get number of cuda gpus available", L"", L"" },
  { L"gpu_cudaGetLastErrorString", L"Q", L"gpu_cudaGetLastErrorString", L"", L"1", L"", L"", L"", L"Get last CUDA Error Text", L"", L"" },
  { L"gpu_train_numeric", L"QK%K%K%K%JJJQ", L"gpu_train_numeric", L"input,target,widths,transfns,nIter,retMode,useSuppliedInitialW0,W0", L"1", L"", L"", L"", L"Train a neural net", L"Input Layer with intercept (1,X)", L"Target Layer (Y_obs)", L"Layer widths", L"Transfer functions and Loss function", L"epochs to train", L"return Mode", L"0 for no, nonzero for yes", L"Initial weights" },
  { L"gpu_totalGlobalMemInMB", L"Q", L"gpu_totalGlobalMemInMB", L"", L"1", L"", L"", L"", L"Query gpu memory" },
  { L"gpu_cudaVersion", L"Q", L"gpu_cudaVersion", L"", L"1", L"", L"", L"", L"Query CUDA Version" },
  { L"gpu_name", L"Q", L"gpu_name", L"", L"1", L"", L"", L"", L"Query gpu name" },
  { L"gpu_cudaMalloc", L"QK%", L"gpu_cudaMalloc", L"arr", L"1", L"", L"", L"", L"allocate gpu memory and get address" },
  { L"gpu_cudaFree", L"QC%", L"gpu_cudaFree", L"HexAddr", L"1", L"", L"", L"", L"free gpu memory at input address" },
  //{ L"gpu_getGpuData", L"QC%JJ", L"gpu_getGpuData", L"HexAddr,nrow,ncol", L"1", L"", L"", L"", L"" },
  { L"gpu_cublasDscal", L"QK%BJJ", L"gpu_cublasDscal", L"x,alpha,incx,n", L"1", L"", L"", L"", L"rescale array", L"array to be rescaled", L"the scalar", L"(optional) stride between consecutive elements (default 1 as normal)", L"(optional) array length (default 0 for auto determine)" },
  { L"gpu_cublasIdamax", L"QK%JJ", L"gpu_cublasIdamax", L"arr,incx,n", L"1", L"", L"", L"", L"returns index of element with max abs(element)", L"array to be searched", L"(optional) stride between consecutive elements (default 1 as normal)", L"(optional) array length (default 0 for auto determine)" },
  { L"gpu_cublasIdamin", L"QK%JJ", L"gpu_cublasIdamin", L"arr,incx,n", L"1", L"", L"", L"", L"returns index of element with min abs(element)", L"array to be searched", L"(optional) stride between consecutive elements (default 1 as normal)", L"(optional) array length (default 0 for auto determine)" },
  { L"gpu_cublasDasum", L"QK%JJ", L"gpu_cublasDasum", L"arr,incx,n", L"1", L"", L"", L"", L"sum(abs(arr))", L"array to sum", L"(optional) stride between consecutive elements (default 1 as normal)", L"(optional) array length (default 0 for auto determine)" },
  { L"gpu_cublasDaxpy", L"QBK%K%JJJ", L"gpu_cublasDaxpy", L"a,X,Y,incx,incy,n", L"1", L"", L"", L"", L"aX+Y", L"scalar", L"array", L"array",L"X stride", L"Y stride", L"actual #partipated elem of both X & Y" },
  { L"gpu_cublasDdot", L"QK%K%JJJ", L"gpu_cublasDdot", L"X,Y,incx,incy,n", L"1", L"", L"", L"", L"dotproduct(X,Y)", L"x", L"y", L"X stride", L"Y stride", L"actual #partipated elem of both X & Y" },
  { L"gpu_cublasDnrm2", L"QK%JJ", L"gpu_cublasDnrm2", L"X,incx,n", L"1", L"", L"", L"", L"L2 norm", L"X", L"X stride", L"actual #partipated elem of X" },
  { L"gpu_cublasDgemm", L"QBK%JK%JBK%JJJJJJ", L"gpu_cublasDgemm", L"alpha,A,transa,B,transb,beta,C,m,k,n,lda,ldb,ldc", L"1", L"", L"", L"", L"alpha*op(A)*op(B)+beta*C",L"scalar", L"matrix", L"0-no, 1-yes", L"matrix", L"0-no, 1-yes", L"scalar", L"matrix", L"nrow op(A)", L"ncol op(A)", L"ncol op(B)", L"ncol A", L"ncol B", L"nrow C  " },
  { L"gpu_cublasDsymm", L"QBK%JJK%BK%JJJJJ", L"gpu_cublasDsymm", L"alpha,A,uplo,side,B,beta,C,m,n,lda,ldb,ldc", L"1", L"", L"", L"", L"alpha*AB+beta*C or alpha*BA+beta*C",L"scalar", L"matrix", L"0-AB, 1-BA", L"0-lo, 1-up", L"matrix", L"scalar", L"matrix", L"nrow C and B", L"ncol C and B", L"ncol op(B)", L"ncol A", L"ncol B", L"ncol C  " },
  { L"gpu_cublasDsyrk", L"QBK%JBK%JJJJJ", L"gpu_cublasDsyrk", L"alpha,A,trans,beta,C,uplo,n,k,lda,ldc", L"1", L"", L"", L"", L"alpha*op(A)op(A)'+beta*C",L"scalar", L"matrix", L"0-no, 1-yes", L"scalar", L"matrix", L"0-lo, 1-up (for C)", L"nrow op(A) and C", L"ncol op(A)", L"ncol A", L"ncol C  " },
  { L"gpu_cublasDgeam", L"QBK%JBK%JJJJJ", L"gpu_cublasDgeam", L"alpha,A,transa,beta,B,transb,m,n,lda,ldb", L"1", L"", L"", L"", L"alpha*op(A)+beta*op(B)",L"scalar", L"matrix", L"0-no, 1-yes",L"scalar", L"matrix", L"0-no, 1-yes",L"nrow op(A) and output", L"ncol op(B) and output", L"ncol A", L"ncol B  " },
  { L"gpu_cublasDdgmm", L"QK%K%JJJJJ", L"gpu_cublasDdgmm", L"x,A,m,n,mode,incx,lda", L"1", L"", L"", L"", L"diag(x)*A or A * diag(x)",L"vector", L"matrix",L"nrow A and output", L"ncol A and output", L"0-diag(x) on the left, 1-diag(x) on the right", L"x stride", L"ncol A " },
  { L"gpu_inv", L"QK%", L"gpu_inv", L"mat", L"1", L"", L"", L"", L"matrix inversion", L"matrix" },
  { L"gpu_cusolverDnDsyevd", L"QK%JJJ", L"gpu_cusolverDnDsyevd", L"symat,uplo,jobz,eigvalDesc", L"1", L"", L"", L"", L"eigen decomposition of a real symmetric matrix. Each col returned starts with the eigval, then components of the corresponding eigvec", L"symmetric matrix", L"0-lo, 1-up", L"0-eigval only, 1-eigval and vec", L"0-min eigval first, 1-max eigval first" },
  { L"gpu_eigsym", L"QK%", L"gpu_eigsym", L"symat", L"1", L"", L"", L"", L"eigen decomposition of a real symmetric matrix. Each col returned starts with the eigval, then components of the corresponding eigvec", L"symmetric matrix (uses lower part if not symmetric)" }
  ,{ L"gpu_eig", L"QK%JJ", L"gpu_eig", L"symat,eigvec?,desc?", L"1", L"", L"", L"", L"eigen decomposition of a real matrix. Each col returned starts with the eigval, then components of the corresponding eigvec", L"matrix" }
};

int WINAPI xlAutoOpen(void) {
  static XLOPER12 xllname;
  Excel12f(xlGetName, &xllname, 0);

  for (int i = 0; i < rgFuncsRows; ++i) {
    Excel12f(xlfRegister, 0, rgFuncsCols + 1,
      (LPXLOPER12)&xllname,
      (LPXLOPER12)TempStr12(rgFuncs[i][0]),
      (LPXLOPER12)TempStr12(rgFuncs[i][1]),
      (LPXLOPER12)TempStr12(rgFuncs[i][2]),
      (LPXLOPER12)TempStr12(rgFuncs[i][3]),
      (LPXLOPER12)TempStr12(rgFuncs[i][4]),
      (LPXLOPER12)TempStr12(rgFuncs[i][5]),
      (LPXLOPER12)TempStr12(rgFuncs[i][6]),
      (LPXLOPER12)TempStr12(rgFuncs[i][7]),
      (LPXLOPER12)TempStr12(rgFuncs[i][8]),
      (LPXLOPER12)TempStr12(rgFuncs[i][9]),
      (LPXLOPER12)TempStr12(rgFuncs[i][10]),
      (LPXLOPER12)TempStr12(rgFuncs[i][11]),
      (LPXLOPER12)TempStr12(rgFuncs[i][12]),
      (LPXLOPER12)TempStr12(rgFuncs[i][13]),
      (LPXLOPER12)TempStr12(rgFuncs[i][14]),
      (LPXLOPER12)TempStr12(rgFuncs[i][15]),
      (LPXLOPER12)TempStr12(rgFuncs[i][16]),
      (LPXLOPER12)TempStr12(rgFuncs[i][17]),
      (LPXLOPER12)TempStr12(rgFuncs[i][18]),
      (LPXLOPER12)TempStr12(rgFuncs[i][19]),
      (LPXLOPER12)TempStr12(rgFuncs[i][20]),
      (LPXLOPER12)TempStr12(rgFuncs[i][21]));
  }
  Excel12f(xlFree, 0, 1, (LPXLOPER12)&xllname);
  logfile.open("c:/temp/cuxl.log", ofstream::app);
  logfile << "new session begins" << endl;
  return TRUE;
}

void WINAPI xlAutoFree12(LPXLOPER12 p) { // dummy p
  if (hArray) {
    GlobalUnlock(hArray);
    GlobalFree(hArray);
    hArray = 0;
  }
  logfile << "session ends" << endl;
  logfile.close();
  return;
}

void WINAPI xlAutoAdd(void) {
  XCHAR szBuf[255];
  wsprintf((LPWSTR)szBuf, L"Disclaimer: cuxl.xll is a purely experimental library that comes with no warranty.");
  Excel12f(xlcAlert, 0, 2, TempStr12(szBuf), TempInt12(2));
}

void WINAPI xlAutoRemove(void) {
  if (hArray) {
    GlobalUnlock(hArray);
    GlobalFree(hArray);
    hArray = 0;
  }
  logfile << "session ends" << endl;
  logfile.close();
  return;
}

void WINAPI xlAutoClose(void) {
  if (hArray) {
    GlobalUnlock(hArray);
    GlobalFree(hArray);
    hArray = 0;
  }
  cudaDeviceReset();
  logfile << "session ends" << endl;
  logfile.close();
  return;
}

LPXLOPER12 WINAPI xlAddInManagerInfo12(LPXLOPER12 xAction) {
  static XLOPER12 xInfo, xIntAction;
  Excel12f(xlCoerce, &xIntAction, 2, xAction, TempInt12(xltypeInt));
  if (xIntAction.val.w == 1) {
    xInfo.xltype = xltypeStr;
    xInfo.val.str = L"\030A cuda library for Excel";
  }
  else {
    xInfo.xltype = xltypeErr;
    xInfo.val.err = xlerrValue;
  }

  return (LPXLOPER12)&xInfo;
}

/* ================================================
converters : string, cstring, wchar, hex
================================================ */
XCHAR* pc2pwc(XCHAR* pwc, const char* pc) {
  int len = (int)strlen(pc);
  pwc = (XCHAR*)malloc((len + 1) * sizeof(XCHAR));
  for (int i = 0; i < len; ++i) {
    pwc[i] = (XCHAR)pc[i];
  }
  pwc[len] = 0;
  return pwc;
}

string pwc2str(const XCHAR* pwc) {
  wstring ws(pwc);
  string s(ws.begin(), ws.end());
  return s;
}

XCHAR* str2pwc(XCHAR* pwc, std::string s) {
  int len = (int)s.length();
  pwc = (XCHAR*)malloc((len + 1) * sizeof(XCHAR));
  for (int i = 0; i < len; ++i) {
    pwc[i] = (XCHAR)s.at(i);
  }
  pwc[len] = 0;
  return pwc;
}

void* pc2hex(void* p, const char* s) {
  sscanf_s(s, "%p", &p);
  return p;
}

void* pwc2hex(void* p, const XCHAR* ws) {
  string addr = pwc2str(ws);
  sscanf_s(addr.c_str(), "%p", &p);
  return p;
}

string hex2str(void* p) {
  ostringstream ss;
  ss << p;
  string buf;
  buf = "0x" + ss.str();
  return buf;
}

char* hex2pc(char* pc, void* p) {
  strcpy(pc, hex2str(p).c_str());
  return pc;
}

XCHAR* hex2pwc(XCHAR* pwc, void* p) {
  pwc = str2pwc(pwc, hex2str(p));
  return pwc;
}

void alloc_cpu(double** a, double** da, magma_int_t m) {
  magma_dmalloc_cpu(a, m);
  if (da) magma_dmalloc(da, m);
}
void free_cpu(double** a, double** da) {
  if (da) magma_free(*da);
  magma_free_cpu(*a);
}
void alloc_pinned(double** a, double** da, magma_int_t m) {
  magma_dmalloc_pinned(a, m);
  if (da) magma_dmalloc(da, m);
}
void free_pinned(double** a, double** da) {
  if (da) magma_free(*da);
  magma_free_pinned(*a);
}

BOOL APIENTRY DllMain(HMODULE hModule,
  DWORD ul_reason_for_call,
  LPVOID lpReserved
) {
  switch (ul_reason_for_call) {
  case DLL_PROCESS_ATTACH:
  case DLL_THREAD_ATTACH:
  case DLL_THREAD_DETACH:
  case DLL_PROCESS_DETACH:
    //lesson hard learned: do not place cudaDeviceReset() here, it will freeze Excel when doing cudaMalloc
    break;
  }
  return TRUE;
}

/* =========================================
cuda
========================================= */

LPXLOPER12 WINAPI gpu_cudaDeviceReset() {
  return TempInt12(cudaDeviceReset());
}

LPXLOPER12 WINAPI gpu_cudaGetLastError() {
  return TempInt12(cudaGetLastError());
}

LPXLOPER12 WINAPI gpu_cudaGetLastErrorString() {
  XCHAR* s = nullptr;
  s = pc2pwc(s, cudaGetErrorString(cudaGetLastError()));
  return TempStr12(s);
}

LPXLOPER12 WINAPI gpu_cudaSetDevice(int dev) {
  return TempInt12(cudaSetDevice(dev));
}

LPXLOPER12 WINAPI gpu_cudaGetDeviceCount() {
  int count;
  if (cudaSuccess != cudaGetDeviceCount(&count)) {
    return NULL;
  }
  else {
    return TempInt12(count);
  }
}

LPXLOPER12 WINAPI gpu_totalGlobalMemInMB() {
  int iDev;
  cuAsrt(lbl_error, cudaGetDevice(&iDev));
  cudaDeviceProp prop;
  cuAsrt(lbl_error, cudaGetDeviceProperties(&prop, iDev));
  return TempInt12((int)(prop.totalGlobalMem / 1024 / 1024));

lbl_error:
  XCHAR* errmsg = nullptr;
  errmsg = pc2pwc(errmsg, cudaGetErrorString(cudaErrorCode));
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cudaVersion() {
  int iDev;
  cuAsrt(lbl_error, cudaGetDevice(&iDev));
  cudaDeviceProp prop;
  cuAsrt(lbl_error, cudaGetDeviceProperties(&prop, iDev));
  return TempNum12(prop.major + prop.minor * 0.1);
lbl_error:
  XCHAR* errmsg = nullptr;
  errmsg = pc2pwc(errmsg, cudaGetErrorString(cudaErrorCode));
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_name() {
  cudaError_t cudaErrorCode;
  int iDev;
  cuAsrt(lbl_error, cudaGetDevice(&iDev));
  cudaDeviceProp prop;
  cuAsrt(lbl_error, cudaGetDeviceProperties(&prop, iDev));
  XCHAR* s = nullptr;
  s = pc2pwc(s, prop.name);
  return TempStr12(s);
lbl_error:
  XCHAR* errmsg = nullptr;
  errmsg = pc2pwc(errmsg, cudaGetErrorString(cudaErrorCode));
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cudaMalloc(FP12 x) {
  long nrow = x.rows;
  long ncol = x.columns;
  long n = nrow * ncol;
  double* a = x.array;
  float* b = (float*)calloc(n, sizeof(float));
  for (long i = 0; i < n; ++i) {
    b[i] = (float)(a[i]);
  }
  void* db = 0;
  string buf;
  stringstream ss;
  cuAsrt(lbl_error, cudaMalloc(&db, n * sizeof(float)));
  cuAsrt(lbl_error, cudaMemcpy(db, b, n, cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  free(b);

  ss << db;
  buf = "0x" + ss.str();
  XCHAR* s = nullptr;
  s = str2pwc(s, buf);
  return TempStr12(s);
lbl_error:
  if (!b) free(b);
  if (!db) cudaFree(db);
  XCHAR* errmsg = nullptr;
  errmsg = pc2pwc(errmsg, cudaGetErrorString(cudaErrorCode));
  errmsg = pc2pwc(errmsg, "");
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cudaFree(XCHAR* str_d_x) {
  void* d_x = 0;
  d_x = pwc2hex(d_x, str_d_x);
  XCHAR* errmsg = nullptr;
  errmsg = pc2pwc(errmsg, cudaGetErrorString(cudaFree(d_x)));
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_getGpuData(XCHAR* str_d_x, int m, int n) {
  void* d_x = 0;
  d_x = pwc2hex(d_x, str_d_x);
  double* res = (double*)calloc(m * n, sizeof(double));
  cudaMemcpy(res, d_x, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  static XLOPER12 xMulti;
  //initialize xMulti attributes
  xMulti.xltype = xltypeMulti | xlbitDLLFree;
  xMulti.val.array.rows = m;
  xMulti.val.array.columns = n;
  //allocate host global memory for xMulti.val.array.lparray
  xMulti.val.array.lparray = (LPXLOPER12)GlobalLock(hArray = GlobalAlloc(GMEM_ZEROINIT, m * n * sizeof(XLOPER12)));
  double elem;
  for (int i = 0; i < m*n; ++i) {
    elem = res[i];
    xMulti.val.array.lparray[i].val.num = elem;
    xMulti.val.array.lparray[i].xltype = xltypeNum;
  }
  free(res);
  return (LPXLOPER12)(&xMulti);
lbl_error:
  free(res);
}



/* =================================================
tests
================================================= */
__global__ void test_kern(int leng, float* x, float* y) {
  int i = threadIdx.x;
  if (i < leng) {
    y[i] = x[i] * 3.;
  }
}


LPXLOPER12 WINAPI gpu_test(FP12 x) {
  logfile << endl << "new gpu tests" << endl;

  logfile << "cudaDeviceReset" << endl;
  cudaDeviceReset();
  float* d_x, *d_y;
  logfile << "cudaMalloc" << endl;
  cudaMalloc((void**)&d_x, 10 * sizeof(float));
  cudaMalloc((void**)&d_y, 10 * sizeof(float));

  logfile << "kernel launch" << endl;
  test_kern << <1, 16 >> >(10, d_x, d_y);

  logfile << "cudaFree" << endl;
  cudaFree(d_x);
  cudaFree(d_y);

  logfile << "double->float" << endl;
  void* p = x.array;
  int n = x.rows * x.columns;
  for (int i = 0; i < n * 2; i += 2) {
    logfile << (i / 2)[(double*)p] << " float cast: " << static_cast<float>((i / 2)[(double*)p]) << ";  float*:" << i[(float*)p] << ", " << (i + 1)[(float*)p] << endl;
  }

  logfile << "cudaDeviceReset" << endl;
  cudaDeviceReset();

  return TempStr12(L"done");
}


/* ==============================================
shared: cuBlas, cuSolver
===============================================*/

const char* cusolverGetErrorString(cusolverStatus_t status) {

  switch (status)
  {
  case CUSOLVER_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
  case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
  case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
  case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
  case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
  case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
  case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
  case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  }
  return "cusolver unknown error";
}

string get_errmsg_str() {
  string errmsg_str;
  switch (errortype)
  {
  case 1:
    errmsg_str = cudaGetErrorString(cudaErrorCode);
    break;
  case 2:
    errmsg_str = cublasGetErrorString(cublasErrorCode);
    break;
  case 3:
    errmsg_str = cusolverGetErrorString(cusolverErrorCode);
    break;
  default:
    errmsg_str = "unknown error";
    break;
  }
  return errmsg_str;
}

const char* cublasGetErrorString(cublasStatus_t status)
{
  switch (status)
  {
  case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "cublas unknown error";
}

bool inline default_missing_n_inc(int& n, int& inc, const int leng) {
  if (inc <= 0) inc = 1;
  int m = leng / inc;
  if (n <= 0) n = m;
  return (n == m);
}

class cuBagBase {
public:
  int nx() { return (incx > 0 ? (lengx - 1) / incx + 1 : 0); }
  int ny() { return (incy > 0 ? (lengy - 1) / incy + 1 : 0); }
  int lengx;
  int lengy;
  int n;
  int m;
  int k;
  double* d_x = 0;
  double* d_y = 0;
  int incx;
  int incy;
  double alpha;
  double beta;
  double c;
  double s;
  double* param = 0;
  double* d_A = 0;
  double* d_B = 0;
  double* d_C = 0;
  int lda;
  int ldb;
  int ldc;
  cublasOperation_t transa;
  cublasOperation_t transb;
  cublasSideMode_t side;
  cublasFillMode_t uplo;
  int batchSize;
  int Lwork;
  double* d_work;


  bool alloc_d_x(FP12& src) {
    lengx = src.rows * src.columns;
    if (incx <= 0) incx = 1;
    if (n <= 0) n = nx();
    asrt(lbl_error, L"length.out error", n == nx());
    cuAsrt(lbl_error, cudaMalloc((void**)&d_x, lengx * sizeof(*src.array)));

    return TRUE;
  lbl_error:
    return FALSE;
  }

  bool alloc_d_y(FP12& src) {
    lengy = src.rows * src.columns;
    if (incy <= 0) incy = 1;
    if (n <= 0) n = ny();
    asrt(lbl_error, L"length.out error", n == ny());
    cuAsrt(lbl_error, cudaMalloc((void**)&d_y, lengy * sizeof(*src.array)));

    return TRUE;
  lbl_error:
    return FALSE;
  }

  bool alloc_d_A(FP12& src) {
    cuAsrt(lbl_error, cudaMalloc((void**)&d_A, m * k * sizeof(*src.array)));

    return TRUE;
  lbl_error:
    return FALSE;
  }

  bool alloc_d_B(FP12& src) {
    cuAsrt(lbl_error, cudaMalloc((void**)&d_B, k * n * sizeof(*src.array)));

    return TRUE;
  lbl_error:
    return FALSE;
  }

  bool alloc_d_C(FP12& src) {
    cuAsrt(lbl_error, cudaMalloc((void**)&d_C, m * n * sizeof(double)));

    return TRUE;
  lbl_error:
    return FALSE;
  }

  bool alloc_d_work() {
    cuAsrt(lbl_error, cudaMalloc((void**)&d_work, Lwork * sizeof(double)));
    return TRUE;
  lbl_error:
    return FALSE;
  }

  cuBagBase() {
    //pass
  }

  ~cuBagBase() {
    //always free/destroy
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);
    if (param) cudaFree(param);
    if (d_work) cudaFree(d_work);
  }
};

struct hcublas {
  cublasHandle_t m_handle;
  bool live;

  cublasHandle_t handle() {
    if (live) {
      return m_handle;
    }
    else {
      return 0;
    }

  }
  cublasStatus_t create() {

    if (live) destroy();

    cublasErrorCode = cublasCreate(&m_handle);
    if (CUBLAS_STATUS_SUCCESS == cublasErrorCode) {
      live = TRUE;
    }
    return cublasErrorCode;

  }

  cublasStatus_t destroy() {

    if (live) {
      cublasErrorCode = cublasDestroy(m_handle);
      if (CUBLAS_STATUS_SUCCESS == cublasErrorCode) {
        live = FALSE;
      }
      return cublasErrorCode;
    }
    return CUBLAS_STATUS_SUCCESS;
  }
} hcublas;

class cublasBag : public cuBagBase {
public:
  cublasBag() {
    hcublas.create();
  }

  ~cublasBag() {
    hcublas.destroy();
  }
};

struct hcusolverDn {
  cusolverDnHandle_t m_handle;
  bool live;

  cusolverDnHandle_t handle() {
    if (live) {
      return m_handle;
    }
    else {
      return 0;
    }

  }
  cusolverStatus_t create() {

    if (live) destroy();

    cusolverErrorCode = cusolverDnCreate(&m_handle);
    if (CUSOLVER_STATUS_SUCCESS == cusolverErrorCode) {
      live = TRUE;
    }
    return cusolverErrorCode;

  }

  cusolverStatus_t destroy() {

    if (live) {
      cusolverErrorCode = cusolverDnDestroy(m_handle);
      if (CUSOLVER_STATUS_SUCCESS == cusolverErrorCode) {
        live = FALSE;
      }
      return cusolverErrorCode;
    }
    return CUSOLVER_STATUS_SUCCESS;
  }
} hcusolverDn;

class cusolverBag : public cuBagBase {
public:
  int bufferSize = 0;

  int* d_PivotArray = 0;
  int* d_infoArray = 0;
  int* d_info = 0;
  int h_info = 0;
  double* d_buffer = 0;
  int* d_ipiv = 0;
  cusolverEigMode_t jobz;
  cusolverEigType_t itype;
  double* d_W = 0;  //a real array of dimension n. The eigenvalues of A, sorted so that W(i) >= W(i+1).


  bool init_d_info() {
    cuAsrt(lbl_error, cudaMalloc(&d_info, sizeof(int)));
    cuAsrt(lbl_error, cudaMemset(d_info, 0, sizeof(int)));
    return TRUE;
  lbl_error:
    return FALSE;
  }

  bool alloc_buffer_ipiv() {
    cuAsrt(lbl_error, cudaMalloc(&d_buffer, sizeof(double) * bufferSize));
    cuAsrt(lbl_error, cudaMalloc(&d_ipiv, sizeof(int) * n));
    return TRUE;
  lbl_error:
    return FALSE;
  }

  bool fetch_h_info() {
    cuAsrt(lbl_error, cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    return TRUE;
  lbl_error:
    return FALSE;
  }

  bool init_identity_d_B() {
    double* B = (double*)calloc(n * n, sizeof(double));
    for (int i = 0; i < n * n; i += n + 1) B[i] = 1.;
    cuAsrt(lbl_error, cudaMalloc((void**)&d_B, n * n * sizeof(double)));
    cuAsrt(lbl_error, cudaMemcpy(d_B, B, n * n * sizeof(double), cudaMemcpyHostToDevice));
    free(B);
    return TRUE;
  lbl_error:
    free(B);
    errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
    return FALSE;
  }

  bool init_d_A(FP12& src) {
    asrt(lbl_error, L"failed allocating d_A", alloc_d_A(src));
    cuAsrt(lbl_error, cudaMemcpy(d_A, (src.array), n * n * sizeof(double), cudaMemcpyHostToDevice));
    return TRUE;
  lbl_error:
    return FALSE;
  }

  bool init_d_C() {
    cuAsrt(lbl_error, cudaMalloc((void**)&d_C, n * n * sizeof(double)));
    cuAsrt(lbl_error, cudaMemcpy(d_C, d_B, n * n * sizeof(double), cudaMemcpyDeviceToDevice));
    return TRUE;
  lbl_error:
    return FALSE;
  }

  bool alloc_d_W() {
    cuAsrt(lbl_error, cudaMalloc((void**)&d_W, n * sizeof(double)));
    return TRUE;
  lbl_error:
    return FALSE;
  }


  cusolverBag() {
    hcusolverDn.create();
  }

  ~cusolverBag() {
    //always free/destroy
    hcusolverDn.destroy();
    if (d_B) cudaFree(d_B);
    if (d_W) cudaFree(d_W);
    if (d_PivotArray) cudaFree(d_PivotArray);
    if (d_infoArray) cudaFree(d_infoArray);
  }
};

LPXLOPER12 WINAPI gpu_cublasGetLastHandle() {
  XCHAR* pwc = nullptr;
  pwc = hex2pwc(pwc, (void*)hcublas.handle());
  return TempStr12(pwc);
}

LPXLOPER12 WINAPI gpu_cublasCreate() {
  hcublas.create();
  return gpu_cublasGetLastHandle();
}

LPXLOPER12 WINAPI gpu_cublasDestroy() {
  hcublas.destroy();
lbl_error:
  XCHAR* errmsg = nullptr;
  errmsg = pc2pwc(errmsg, cublasGetErrorString(cublasErrorCode));
  return TempStr12(errmsg);
}

/* ==============================================
cublas level 1
============================================== */
LPXLOPER12 WINAPI gpu_cublasIdamax(FP12& x, int incx, int n) {
  //returns index of element with max abs(element)
  int res;
  cublasBag r = cublasBag();
  r.incx = incx;
  r.n = n;

  asrt(lbl_error, L"error acquring device x", r.alloc_d_x(x));

  cublasAsrt(lbl_error, cublasSetVector(r.n, sizeof(*x.array), x.array, r.incx, r.d_x, r.incx));
  cublasAsrt(lbl_error, cublasIdamax(hcublas.handle(), r.n, r.d_x, r.incx, &res));

  return TempInt12(res);
lbl_error:
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cublasIdamin(FP12& x, int incx, int n) {
  //returns index of element with min abs(element)
  int res;
  cublasBag r = cublasBag();
  r.incx = incx;
  r.n = n;

  asrt(lbl_error, L"error acquring device x", r.alloc_d_x(x));

  cublasAsrt(lbl_error, cublasSetVector(r.n, sizeof(*x.array), x.array, r.incx, r.d_x, r.incx));
  cublasAsrt(lbl_error, cublasIdamin(hcublas.handle(), r.n, r.d_x, r.incx, &res));

  return TempInt12(res);
lbl_error:
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cublasDasum(FP12& x, int incx, int n) {
  //sum(abs(x))
  double res;
  cublasBag r = cublasBag();
  r.incx = incx;
  r.n = n;

  asrt(lbl_error, L"error acquring device x", r.alloc_d_x(x));

  cublasAsrt(lbl_error, cublasSetVector(r.n, sizeof(*x.array), x.array, r.incx, r.d_x, r.incx));
  cublasAsrt(lbl_error, cublasDasum(hcublas.handle(), r.n, r.d_x, r.incx, &res));

  return TempNum12(res);
lbl_error:
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cublasDaxpy(double alpha, FP12& x, FP12& y, int incx, int incy, int n) {
  //returns index of element with max abs(element)  
  XCHAR* errmsg = nullptr;
  //regularize n and incx  
  int lengx = x.rows * x.columns;
  asrt(lbl_return_errmsg, L"incorrect length.out nx", default_missing_n_inc(n, incx, lengx));

  int ny = 0;
  int lengy = y.rows * y.columns;
  asrt(lbl_return_errmsg, L"incorrect length.out ny", default_missing_n_inc(ny, incy, lengy));

  asrt(lbl_return_errmsg, L"unequal length.out between x and y", n == ny);

  //gpu memory
  double* d_x;
  cuAsrt(lbl_error, cudaMalloc((void**)&d_x, lengx * sizeof(*x.array)));
  double* d_y;
  cuAsrt(lbl_error, cudaMalloc((void**)&d_y, lengy * sizeof(*y.array)));
  double* res = (double*)calloc(lengy, sizeof(*y.array));
  hcublas.create();
  cublasAsrt(lbl_error, cublasSetVector(n, sizeof(*x.array), x.array, incx, d_x, incx));
  cublasAsrt(lbl_error, cublasSetVector(n, sizeof(*y.array), y.array, incx, d_y, incx));
  cublasAsrt(lbl_error, cublasDaxpy(hcublas.handle(), n, &alpha, d_x, incx, d_y, incy));
  cublasAsrt(lbl_error, cublasGetVector(n, sizeof(*y.array), d_y, incy, res, incy));
  hcublas.destroy();
  cudaFree(d_x);
  cudaFree(d_y);

  static XLOPER12 xMulti;
  //initialize xMulti attributes
  xMulti.xltype = xltypeMulti | xlbitDLLFree;
  xMulti.val.array.rows = y.rows;
  xMulti.val.array.columns = y.columns;
  //allocate host global memory for xMulti.val.array.lparray
  xMulti.val.array.lparray = (LPXLOPER12)GlobalLock(hArray = GlobalAlloc(GMEM_ZEROINIT, lengy * sizeof(XLOPER12)));
  for (int i = 0; i < lengy; ++i) {
    xMulti.val.array.lparray[i].val.num = res[i];
    xMulti.val.array.lparray[i].xltype = xltypeNum;
  }
  free(res);

  return (LPXLOPER12)(&xMulti);
lbl_error:
  hcublas.destroy();
  cudaFree(d_x);
  cudaFree(d_y);
  free(res);
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cublasDscal(FP12& x, double alpha, int incx, int n) {
  double* res = 0;
  cublasBag r = cublasBag();
  r.alpha = alpha;
  r.incx = incx;
  r.n = n;
  asrt(lbl_error, L"error acquiring d_x", r.alloc_d_x(x));
  cublasAsrt(lbl_error, cublasSetVector(r.n, sizeof(*x.array), x.array, r.incx, r.d_x, r.incx));
  cublasAsrt(lbl_error, cublasDscal(hcublas.handle(), r.lengx, &r.alpha, r.d_x, r.incx));
  res = (double*)calloc(r.lengx, sizeof(*x.array));
  cublasAsrt(lbl_error, cublasGetVector(r.n, sizeof(*x.array), r.d_x, r.incx, res, r.incx));
  static XLOPER12 xMulti;
  //initialize xMulti attributes
  xMulti.xltype = xltypeMulti | xlbitDLLFree;
  xMulti.val.array.rows = x.rows;
  xMulti.val.array.columns = x.columns;
  //allocate host global memory for xMulti.val.array.lparray
  xMulti.val.array.lparray = (LPXLOPER12)GlobalLock(hArray = GlobalAlloc(GMEM_ZEROINIT, r.lengx * sizeof(XLOPER12)));
  for (int i = 0; i < r.lengx; ++i) {
    xMulti.val.array.lparray[i].val.num = res[i];
    xMulti.val.array.lparray[i].xltype = xltypeNum;
  }
  free(res);
  return (LPXLOPER12)(&xMulti);
lbl_error:
  free(res);
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cublasDdot(FP12& x, FP12& y, int incx, int incy, int n) {
  double res;
  cublasBag r = cublasBag();
  r.incx = incx;
  r.incy = incy;
  r.n = n;

  asrt(lbl_error, L"error acquiring d_x", r.alloc_d_x(x));
  asrt(lbl_error, L"error acquiring d_y", r.alloc_d_y(y));
  cublasAsrt(lbl_error, cublasSetVector(n, sizeof(*x.array), x.array, r.incx, r.d_x, r.incx));
  cublasAsrt(lbl_error, cublasSetVector(n, sizeof(*y.array), y.array, r.incy, r.d_y, r.incy));
  cublasAsrt(lbl_error, cublasDdot(hcublas.handle(), r.n, r.d_x, r.incx, r.d_y, r.incy, &res));
  return TempNum12(res);
lbl_error:
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cublasDnrm2(FP12& x, int incx, int n) {
  double res;
  cublasBag r = cublasBag();
  r.incx = incx;
  r.n = n;
  asrt(lbl_error, L"error acquiring d_x", r.alloc_d_x(x));
  cublasAsrt(lbl_error, cublasSetVector(n, sizeof(*x.array), x.array, r.incx, r.d_x, r.incx));
  cublasAsrt(lbl_error, cublasDnrm2(hcublas.handle(), r.n, r.d_x, r.incx, &res));
  return TempNum12(res);
lbl_error:
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

/* ================================================
cublas level 3
================================================ */
LPXLOPER12 WINAPI gpu_cublasDgemm(double alpha, FP12& A, int transa, FP12& B, int transb, double beta, FP12& C, int m, int k, int n, int lda, int ldb, int ldc) {
  cublasBag r = cublasBag();
  r.alpha = alpha;
  r.beta = beta;
  r.m = m;
  r.k = k;
  r.n = n;
  r.lda = lda;
  r.ldb = ldb;
  r.ldc = ldc;
  r.transa = (cublasOperation_t)transa; // (transa == 0 ? 1 : 0);
  r.transb = (cublasOperation_t)transb; // (transb == 0 ? 1 : 0);
  double* res = (double*)calloc(r.m*r.n, sizeof(double));
  //memcpy_s((void*)res, r.m*r.n*sizeof(*C.array), (void*)C.array, m*n*sizeof(*C.array));
  asrt(lbl_error, L"error acquiring d_A", r.alloc_d_A(A));
  asrt(lbl_error, L"error acquiring d_B", r.alloc_d_B(B));
  asrt(lbl_error, L"error acquiring d_C", r.alloc_d_C(C));
  cublasAsrt(lbl_error, cublasSetVector(r.m * r.k, sizeof(*A.array), A.array, 1, r.d_A, 1));
  cublasAsrt(lbl_error, cublasSetVector(r.k * r.n, sizeof(*B.array), B.array, 1, r.d_B, 1));
  cublasAsrt(lbl_error, cublasSetVector(r.m * r.n, sizeof(*C.array), res, 1, r.d_C, 1));
  cublasAsrt(lbl_error, cublasDgemm(hcublas.handle(), r.transb, r.transa, r.n, r.m, r.k, &r.alpha, r.d_B, r.ldb, r.d_A, r.lda, &r.beta, r.d_C, r.ldc));

  cublasGetVector(r.m * r.n, sizeof(double), r.d_C, 1, res, 1);
  static XLOPER12 xMulti;
  //initialize xMulti attributes
  xMulti.xltype = xltypeMulti | xlbitDLLFree;
  xMulti.val.array.rows = r.m;
  xMulti.val.array.columns = r.n;
  //allocate host global memory for xMulti.val.array.lparray
  xMulti.val.array.lparray = (LPXLOPER12)GlobalLock(hArray = GlobalAlloc(GMEM_ZEROINIT, r.m * r.n * sizeof(XLOPER12)));
  for (int i = 0; i < r.m * r.n; ++i) {
    xMulti.val.array.lparray[i].val.num = res[i];
    xMulti.val.array.lparray[i].xltype = xltypeNum;
  }
  free(res);
  return (LPXLOPER12)(&xMulti);
lbl_error:
  free(res);
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cublasDsymm(double alpha, FP12& A, int uplo, int side, FP12& B, double beta, FP12& C, int m, int n, int lda, int ldb, int ldc) {
  cublasBag r = cublasBag();
  r.alpha = alpha;
  r.beta = beta;
  r.m = m;
  r.k = m;
  r.n = n;
  r.lda = lda;
  r.ldb = ldb;
  r.ldc = ldc;
  r.side = (cublasSideMode_t)(side ? 0 : 1);
  r.uplo = (cublasFillMode_t)(uplo ? 0 : 1);
  int nrowA = r.side ? r.m : r.n;
  double* res = (double*)calloc(r.m*r.n, sizeof(double));
  //memcpy_s((void*)res, r.m*r.n * sizeof(*C.array), (void*)C.array, m*n * sizeof(*C.array));
  cuAsrt(lbl_error, cudaMalloc((void**)&r.d_A, nrowA * nrowA * sizeof(double)));
  cuAsrt(lbl_error, cudaMalloc((void**)&r.d_B, r.m * r.n * sizeof(double)));
  cuAsrt(lbl_error, cudaMalloc((void**)&r.d_C, r.m * r.n * sizeof(double)));
  cublasAsrt(lbl_error, cublasSetVector(nrowA * nrowA, sizeof(*A.array), A.array, 1, r.d_A, 1));
  cublasAsrt(lbl_error, cublasSetVector(r.m * r.n, sizeof(*B.array), B.array, 1, r.d_B, 1));
  cublasAsrt(lbl_error, cublasSetVector(r.m * r.n, sizeof(*C.array), res, 1, r.d_C, 1));
  cublasAsrt(lbl_error, cublasDsymm(hcublas.handle(), r.side, r.uplo, r.n, r.m, &r.alpha, r.d_A, r.lda, r.d_B, r.ldb, &r.beta, r.d_C, r.ldc));
  cublasGetVector(r.m * r.n, sizeof(double), r.d_C, 1, res, 1);

  static XLOPER12 xMulti;
  //initialize xMulti attributes
  xMulti.xltype = xltypeMulti | xlbitDLLFree;
  xMulti.val.array.rows = r.m;
  xMulti.val.array.columns = r.n;
  //allocate host global memory for xMulti.val.array.lparray
  xMulti.val.array.lparray = (LPXLOPER12)GlobalLock(hArray = GlobalAlloc(GMEM_ZEROINIT, r.m * r.n * sizeof(XLOPER12)));

  for (int i = 0; i < r.m * r.n; ++i) {
    xMulti.val.array.lparray[i].val.num = res[i];
    xMulti.val.array.lparray[i].xltype = xltypeNum;
  }

  free(res);
  return (LPXLOPER12)(&xMulti);
lbl_error:
  free(res);
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cublasDsyrk(double alpha, FP12& A, int trans, double beta, FP12& C, int uplo, int n, int k, int lda, int ldc) {
  cublasBag r = cublasBag();
  r.alpha = alpha;
  r.beta = beta;
  r.transa = (cublasOperation_t)(trans ? 0 : 1);
  r.k = k;
  r.n = n;
  r.m = n;
  r.lda = lda;
  r.ldc = ldc;
  r.uplo = (cublasFillMode_t)(uplo ? 0 : 1);
  double* res = (double*)calloc(r.m*r.n, sizeof(double));
  cuAsrt(lbl_error, cudaMalloc((void**)&r.d_A, r.n * r.k * sizeof(double)));
  cuAsrt(lbl_error, cudaMalloc((void**)&r.d_C, r.n * r.n * sizeof(double)));
  cublasAsrt(lbl_error, cublasSetVector(r.n * r.k, sizeof(*A.array), A.array, 1, r.d_A, 1));
  cublasAsrt(lbl_error, cublasSetVector(r.n * r.n, sizeof(*C.array), res, 1, r.d_C, 1));
  //cublasAsrt(lbl_error, cublasDsymm(hcublas.handle(), r.side, r.uplo, r.n, r.m, &r.alpha, r.d_A, r.lda, r.d_B, r.ldb, &r.beta, r.d_C, r.ldc));
  cublasAsrt(lbl_error, cublasDsyrk(hcublas.handle(), r.uplo, r.transa, r.n, r.k, &r.alpha, r.d_A, r.lda, &r.beta, r.d_C, r.ldc));
  cublasGetVector(r.m * r.n, sizeof(double), r.d_C, 1, res, 1);

  static XLOPER12 xMulti;
  //initialize xMulti attributes
  xMulti.xltype = xltypeMulti | xlbitDLLFree;
  xMulti.val.array.rows = r.m;
  xMulti.val.array.columns = r.n;
  //allocate host global memory for xMulti.val.array.lparray
  xMulti.val.array.lparray = (LPXLOPER12)GlobalLock(hArray = GlobalAlloc(GMEM_ZEROINIT, r.m * r.n * sizeof(XLOPER12)));
  double elem;
  for (int i = 0; i < r.n; ++i) {
    for (int j = 0; j < r.n; ++j) {
      elem = r.uplo ? res[i * r.n + j] : res[j * r.n + i];
      xMulti.val.array.lparray[i * r.n + j].val.num = elem;
      xMulti.val.array.lparray[j * r.n + i].val.num = elem;
      xMulti.val.array.lparray[i * r.n + j].xltype = xltypeNum;
    }
  }

  free(res);
  return (LPXLOPER12)(&xMulti);
lbl_error:
  free(res);
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

/* =========================================================
cublas ext
========================================================= */
LPXLOPER12 WINAPI gpu_cublasDgeam(double alpha, FP12& A, int transa, double beta, FP12& B, int transb, int m, int n, int lda, int ldb) {
  cublasBag r = cublasBag();
  r.alpha = alpha;
  r.beta = beta;
  r.transa = (cublasOperation_t)(transa);
  r.transb = (cublasOperation_t)(transb);
  r.n = n;
  r.m = m;
  r.lda = lda;
  r.ldb = ldb;
  double* res = (double*)calloc(r.m*r.n, sizeof(double));
  cuAsrt(lbl_error, cudaMalloc((void**)&r.d_A, r.m * r.n * sizeof(double)));
  cuAsrt(lbl_error, cudaMalloc((void**)&r.d_B, r.m * r.n * sizeof(double)));
  cuAsrt(lbl_error, cudaMalloc((void**)&r.d_C, r.m * r.n * sizeof(double)));
  cublasAsrt(lbl_error, cublasSetVector(r.m * r.n, sizeof(*A.array), A.array, 1, r.d_A, 1));
  cublasAsrt(lbl_error, cublasSetVector(r.m * r.n, sizeof(*B.array), B.array, 1, r.d_B, 1));
  cublasAsrt(lbl_error, cublasDgeam(hcublas.handle(), r.transa, r.transb, r.n, r.m, &r.alpha, r.d_A, r.lda, &r.beta, r.d_B, r.ldb, r.d_C, n));
  cublasGetVector(r.m * r.n, sizeof(double), r.d_C, 1, res, 1);

  static XLOPER12 xMulti;
  //initialize xMulti attributes
  xMulti.xltype = xltypeMulti | xlbitDLLFree;
  xMulti.val.array.rows = r.m;
  xMulti.val.array.columns = r.n;
  //allocate host global memory for xMulti.val.array.lparray
  xMulti.val.array.lparray = (LPXLOPER12)GlobalLock(hArray = GlobalAlloc(GMEM_ZEROINIT, r.m * r.n * sizeof(XLOPER12)));
  double elem;
  for (int i = 0; i < r.m*r.n; ++i) {
    elem = res[i];
    xMulti.val.array.lparray[i].val.num = elem;
    xMulti.val.array.lparray[i].xltype = xltypeNum;
  }

  free(res);
  return (LPXLOPER12)(&xMulti);
lbl_error:
  free(res);
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cublasDdgmm(FP12& x, FP12& A, int m, int n, int mode, int incx, int lda) {
  cublasBag r = cublasBag();
  r.side = (cublasSideMode_t)(mode ? 0 : 1);
  r.n = n;
  r.m = m;
  r.incx = incx;
  r.lda = lda;
  r.ldc = r.side ? n : m;

  double* res = (double*)calloc(r.m*r.n, sizeof(double));
  r.lengx = x.rows * x.columns;
  cuAsrt(lbl_error, cudaMalloc((void**)&r.d_x, r.lengx * sizeof(double)));
  cublasAsrt(lbl_error, cublasSetVector(r.side ? m : n, sizeof(double), x.array, r.incx, r.d_x, r.incx));

  cuAsrt(lbl_error, cudaMalloc((void**)&r.d_A, r.m * r.n * sizeof(double)));
  cublasAsrt(lbl_error, cublasSetVector(r.m * r.n, sizeof(*A.array), A.array, 1, r.d_A, 1));

  cuAsrt(lbl_error, cudaMalloc((void**)&r.d_C, r.m * r.n * sizeof(double)));

  cublasAsrt(lbl_error, cublasDdgmm(hcublas.handle(), r.side, r.n, r.m, r.d_A, r.lda, r.d_x, r.incx, r.d_C, r.ldc));
  cublasGetVector(r.m * r.n, sizeof(double), r.d_C, 1, res, 1);

  static XLOPER12 xMulti;
  //initialize xMulti attributes
  xMulti.xltype = xltypeMulti | xlbitDLLFree;
  xMulti.val.array.rows = r.m;
  xMulti.val.array.columns = r.n;
  //allocate host global memory for xMulti.val.array.lparray
  xMulti.val.array.lparray = (LPXLOPER12)GlobalLock(hArray = GlobalAlloc(GMEM_ZEROINIT, r.m * r.n * sizeof(XLOPER12)));
  double elem;
  for (int i = 0; i < r.m*r.n; ++i) {
    elem = res[i];
    xMulti.val.array.lparray[i].val.num = elem;
    xMulti.val.array.lparray[i].xltype = xltypeNum;
  }

  free(res);
  return (LPXLOPER12)(&xMulti);
lbl_error:
  free(res);
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_eigsym(FP12& A);


/* =========================================================
cusolver
========================================================= */

LPXLOPER12 WINAPI gpu_inv(FP12& A) {
  double* res = 0;
  cusolverBag r = cusolverBag();
  r.m = r.k = r.n = A.rows;

  asrt(lbl_return_errmsg, L"not a square matrix for inversion", A.rows == A.columns);

  //A C = I, where I is held by B
  asrt(lbl_error, L"failed initializing d_A on gpu", r.init_d_A(A));
  cusolverAsrt(lbl_error, cusolverDnDgetrf_bufferSize(hcusolverDn.handle(), r.n, r.n, (double*)r.d_A, r.n, &r.bufferSize));

  asrt(lbl_error, L"failed initializing indentity matrix on gpu", r.init_identity_d_B());
  asrt(lbl_error, L"failed initializing d_C on gpu", r.init_d_C());  //C will hold inv(A), initialized copying B
  asrt(lbl_error, L"failed initializing devinfo on the gpu", r.init_d_info());
  asrt(lbl_error, L"failed allocating d_buffer or d_ipiv on the gpu", r.alloc_buffer_ipiv());

  cusolverAsrt(lbl_error, cusolverDnDgetrf(hcusolverDn.handle(), r.n, r.n, r.d_A, r.n, r.d_buffer, r.d_ipiv, r.d_info));
  cusolverAsrt(lbl_error, cusolverDnDgetrs(hcusolverDn.handle(), CUBLAS_OP_N, r.n, r.n, r.d_A, r.n, r.d_ipiv, r.d_C, r.n, r.d_info));
  cuAsrt(lbl_error, cudaDeviceSynchronize());

  res = (double*)calloc(r.n * r.n, sizeof(double));
  cuAsrt(lbl_error, cudaMemcpy(res, r.d_C, r.n * r.n * sizeof(double), cudaMemcpyDeviceToHost));

  static XLOPER12 xMulti;
  //initialize xMulti attributes
  xMulti.xltype = xltypeMulti | xlbitDLLFree;
  xMulti.val.array.rows = r.n;
  xMulti.val.array.columns = r.n;
  //allocate host global memory for xMulti.val.array.lparray
  xMulti.val.array.lparray = (LPXLOPER12)GlobalLock(hArray = GlobalAlloc(GMEM_ZEROINIT, r.n * r.n * sizeof(XLOPER12)));
  double elem;
  for (int i = 0; i < r.n * r.n; ++i) {
    elem = res[i];
    xMulti.val.array.lparray[i].val.num = elem;
    xMulti.val.array.lparray[i].xltype = xltypeNum;
  }
  if (res) free(res);
  return (LPXLOPER12)(&xMulti);
lbl_error:
  if (res) free(res);
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_cusolverDnDsyevd(FP12& A, int uplo, int retEigVec = 1, int eigvalDesc = 0) {
  double* res = 0;
  cusolverBag r = cusolverBag();
  r.jobz = (retEigVec ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR);
  r.lda = r.m = r.k = r.n = A.rows;
  r.uplo = (cublasFillMode_t)(uplo ? 0 : 1);
  asrt(lbl_return_errmsg, L"not a square matrix for inversion", A.rows == A.columns);

  asrt(lbl_error, L"failed initializing d_A on gpu", r.init_d_A(A));
  asrt(lbl_error, L"failed allocating result vector of eigenvalues on gpu", r.alloc_d_W());

  cusolverAsrt(lbl_error, cusolverDnDsyevd_bufferSize(hcusolverDn.handle(), r.jobz, r.uplo, r.n, r.d_A, r.lda, r.d_W, &r.Lwork));
  asrt(lbl_error, L"failed allocating work space on gpu", r.alloc_d_work());
  asrt(lbl_error, L"failed initializing devinfo on gpu", r.init_d_info());
  cusolverAsrt(lbl_error, cusolverDnDsyevd(hcusolverDn.handle(), r.jobz, r.uplo, r.n, r.d_A, r.n, r.d_W, r.d_work, r.Lwork, r.d_info));
  cuAsrt(lbl_error, cudaDeviceSynchronize());

  res = (double*)calloc((retEigVec ? (1 + r.n) : 1) * r.n, sizeof(double));
  cuAsrt(lbl_error, cudaMemcpy(res, r.d_W, r.n * sizeof(double), cudaMemcpyDeviceToHost));
  if (retEigVec) {
    cuAsrt(lbl_error, cudaMemcpy(&res[r.n], r.d_A, r.n * r.n * sizeof(double), cudaMemcpyDeviceToHost));
  }
  static XLOPER12 xMulti;
  //initialize xMulti attributes
  xMulti.xltype = xltypeMulti | xlbitDLLFree;
  xMulti.val.array.rows = (retEigVec ? (1 + r.n) : 1);
  xMulti.val.array.columns = r.n;
  //allocate host global memory for xMulti.val.array.lparray
  xMulti.val.array.lparray = (LPXLOPER12)GlobalLock(hArray = GlobalAlloc(GMEM_ZEROINIT, xMulti.val.array.rows * xMulti.val.array.columns * sizeof(XLOPER12)));
  double elem;
  if (eigvalDesc) { //output is bydefault ascending
    for (int i = 0; i < xMulti.val.array.columns; ++i) {
      elem = res[xMulti.val.array.columns - 1 - i];
      xMulti.val.array.lparray[i].val.num = elem;
      xMulti.val.array.lparray[i].xltype = xltypeNum;
    }
    if (retEigVec) {
      for (int i = xMulti.val.array.columns; i < xMulti.val.array.rows * xMulti.val.array.columns; ++i) {
        elem = res[i];
        xMulti.val.array.lparray[(i % xMulti.val.array.columns + 1) * xMulti.val.array.columns + (xMulti.val.array.columns - i / xMulti.val.array.columns)].val.num = elem;
        xMulti.val.array.lparray[i].xltype = xltypeNum;
      }
    }
  }
  else {
    for (int i = 0; i < xMulti.val.array.columns; ++i) {
      elem = res[i];
      xMulti.val.array.lparray[i].val.num = elem;
      xMulti.val.array.lparray[i].xltype = xltypeNum;
    }
    if (retEigVec) {
      for (int i = xMulti.val.array.columns; i < xMulti.val.array.rows * xMulti.val.array.columns; ++i) {
        elem = res[i];
        xMulti.val.array.lparray[(i % xMulti.val.array.columns + 1) * xMulti.val.array.columns + (i / xMulti.val.array.columns - 1)].val.num = elem;
        xMulti.val.array.lparray[i].xltype = xltypeNum;
      }
    }
  }

  if (res) free(res);
  return (LPXLOPER12)(&xMulti);
lbl_error:
  if (res) free(res);
  errmsg = pc2pwc(errmsg, get_errmsg_str().c_str());
lbl_return_errmsg:
  return TempStr12(errmsg);
}

LPXLOPER12 WINAPI gpu_eigsym(FP12& A) {
  return gpu_cusolverDnDsyevd(A, 0, 1, 1);
}


/* =====================================
X * Y  (matrix multiplication)
===================================== */
bool xy_base(FP12& X, FP12& Y, LPXLOPER12& xMulti, int n, int k, int m, int op) {
  bool ok = false;
  //initialize xMulti attributes
  xMulti = gpu_cublasDgemm(1., X, op >> 1 & 1, Y, op & 1, 0., FP12(), n, k, m, X.columns, Y.columns, m);
  ok = true;
lbl_return:
  return ok;
}

LPXLOPER12 WINAPI gpu_xy(FP12& x, FP12& y) {
  //takes in two floating point arrays
  //outputs reference to the head XLOPER12 in an array of XLOPER12
  bool ok = false;
  int n = x.rows;
  int k = x.columns;
  if (k != y.rows) goto lbl_return;
  int m = y.columns;

  LPXLOPER12 xMulti; //head of result -> should be ok to be static
  ok = xy_base(x, y, xMulti, n, k, m, 0);
  //gpu_cublasDgemm(1.,x,0,y,0,0,0,m,k,n,)
lbl_return:
  if (ok) {
    return xMulti;
  }
  else {
    return NULL;
  };
}

LPXLOPER12 WINAPI gpu_xyt(FP12& x, FP12& y) {
  //takes in two floating point arrays
  //outputs reference to the head XLOPER12 in an array of XLOPER12
  bool ok = false;
  int n = x.rows;
  int k = x.columns;
  if (k != y.columns) goto lbl_return;
  int m = y.rows;

  LPXLOPER12 xMulti; //head of result -> should be ok to be static
  ok = xy_base(x, y, xMulti, n, k, m, 1);

lbl_return:
  if (ok) {
    return xMulti;
  }
  else {
    return NULL;
  }
}

LPXLOPER12 WINAPI gpu_xty(FP12& x, FP12& y) {
  //takes in two floating point arrays
  //outputs reference to the head XLOPER12 in an array of XLOPER12
  bool ok = false;
  int n = x.columns;
  int k = x.rows;
  if (k != y.rows) goto lbl_return;
  int m = y.columns;

  LPXLOPER12 xMulti; //head of result -> should be ok to be static
  ok = xy_base(x, y, xMulti, n, k, m, 2);

lbl_return:
  if (ok) {
    return xMulti;
  }
  else {
    return NULL;
  }
}

LPXLOPER12 WINAPI gpu_xtyt(FP12& x, FP12& y) {
  //takes in two floating point arrays
  //outputs reference to the head XLOPER12 in an array of XLOPER12
  bool ok = false;
  int n = x.columns;
  int k = x.rows;
  if (k != y.columns) goto lbl_return;
  int m = y.rows;

  LPXLOPER12 xMulti; //head of result -> should be ok to be static
  ok = xy_base(x, y, xMulti, n, k, m, 3);

lbl_return:
  if (ok) {
    return xMulti;
  }
  else {
    return NULL;
  }
}

LPXLOPER12 WINAPI gpu_xtx(FP12& x) {
  bool ok = false;
  int n = x.rows;
  int k = x.columns;
  return gpu_cublasDsyrk(1., x, 1, 0., FP12(), 0, k, n, k, k);
}

LPXLOPER12 WINAPI gpu_xxt(FP12& x) {
  bool ok = false;
  int n = x.rows;
  int k = x.columns;
  return gpu_cublasDsyrk(1., x, 0, 0., FP12(), 0, n, k, k, n);
}


/* =====================================
MAGMA
===================================== */

LPXLOPER12 WINAPI  gpu_eig(FP12& A, int noEigVec, int eigvalAsc) {
  int retEigVec = (noEigVec ? 0 : 1);
  magma_int_t lwork, info, i, j, nb, n = A.rows, nsq = n*n;
  double *eigval_r, *eigval_i, *eigvec, *work, *a = A.array;
  nb = magma_get_dgehrd_nb(n);
  lwork = n * (2 + nb);
  lwork = max(lwork, n * (5 + 2 * n));

  magma_dmalloc_cpu(&eigval_r, n);
  magma_dmalloc_cpu(&eigval_i, n);
  magma_dmalloc_cpu(&eigvec, nsq);
  magma_dmalloc_cpu(&work, lwork);

  magma_dgeev((retEigVec ? MagmaVec : MagmaNoVec), MagmaNoVec, n, a, n, eigval_r, eigval_i, eigvec, n, nullptr, n, work, lwork, &info);
  //sort by eigval_r decreasing, eigval_r is mostly sorted descending already
  //check if already decreasing
  int isAlreadyDesc = 1;
  for (int i = 1; i < n; ++i) {
    if (eigval_r[i] > eigval_r[i - 1]) {
      isAlreadyDesc = 0;
      break;
    }
  }
  if (!isAlreadyDesc) {
    //parallel bubble sort

  }

  static XLOPER12 xMulti;
  xMulti.xltype = xltypeMulti | xlbitDLLFree;
  xMulti.val.array.rows = (retEigVec ? (2 + n) : 2);
  xMulti.val.array.columns = n;
  //allocate host global memory for xMulti.val.array.lparray
  xMulti.val.array.lparray = (LPXLOPER12)GlobalLock(hArray = GlobalAlloc(GMEM_ZEROINIT, xMulti.val.array.rows * xMulti.val.array.columns * sizeof(XLOPER12)));
  double elem;
  if (eigvalAsc) { //output is bydefault dscending
    for (int i = 0; i < n; ++i) {
      //real part
      elem = eigval_r[n - 1 - i];
      xMulti.val.array.lparray[i].val.num = elem;
      xMulti.val.array.lparray[i].xltype = xltypeNum;
      //imag part
      elem = eigval_i[n - 1 - i];
      xMulti.val.array.lparray[i + n].val.num = elem;
      xMulti.val.array.lparray[i + n].xltype = xltypeNum;
    }
    if (retEigVec) {
      for (int i = n * 2; i < xMulti.val.array.rows * n; ++i) {
        elem = eigvec[i - 2 * n];
        xMulti.val.array.lparray[(i % n + 2) * n + (n - i / n) + 1].val.num = elem; //the +1 in the index is because i now starts from 2n, not 1n
        xMulti.val.array.lparray[i].xltype = xltypeNum;
      }
    }
  }
  else {
    for (int i = 0; i < n; ++i) {
      
      //real part
      elem = eigval_r[i];
      xMulti.val.array.lparray[i].val.num = elem;
      xMulti.val.array.lparray[i].xltype = xltypeNum;
      //imag part
      elem = eigval_i[i];
      xMulti.val.array.lparray[i + n].val.num = elem;
      xMulti.val.array.lparray[i + n].xltype = xltypeNum;
    }
    if (retEigVec) {
      for (int i = n * 2; i < xMulti.val.array.rows * n; ++i) {
        elem = eigvec[i - 2 * n];
        xMulti.val.array.lparray[(i % n + 2) * n + (i / n - 2)].val.num = elem; //the -2 in the index is because i now starts from 2n not 1n
        xMulti.val.array.lparray[i].xltype = xltypeNum;
      }
    }
  }

  magma_free_cpu(work);
  magma_free_cpu(eigvec);
  magma_free_cpu(eigval_i);
  magma_free_cpu(eigval_r);

  return (LPXLOPER12)&xMulti;
}
