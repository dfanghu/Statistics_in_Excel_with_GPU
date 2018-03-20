#pragma once
/* file comment: defining functions in header file is not preferable as it cause redefinition each time it is included
But this is the only "inline" way to have these code included to the main kernel.cu
Putting only declarations here and opening another .cu to hold the definitions results in a mysterious compilation error.
*/
#include <math.h>
#include "gpu_train.cuh"
/*  index of transfer and loss functions
Category...Function.......1st deriv fn....Index
getter.....getTransFn
getter.....getLossFn
transfer...logit..........dlogit..........0
transfer...tanh...........dtanh...........1
transfer...relu...........drelu...........2
transfer...iden...........diden...........3
loss.......softmax
loss.......sseLoss........dSseLoss........0
loss.......xenLoss........dXenLoss........1

*/

__device__ inline double logit(double x) {
  return tanh(x / 2.) / 2. + 0.5;
}
__device__ inline double dlogit(double y) {
  //y is not x but logit(x)
  return y * (1. - y);
}


//__device__ inline double tanh(double x)  //already in the cmath header, no need to implement
__device__ inline double dtanh(double y) {
  //y is not x but tanh(y)
  return 1. - y * y;
}
__device__ inline double relu(double x) {
  if (fabs(x - 999.) < 1e-16) return 1.;
  return (x > 0) ? x : 0.;
}
__device__ inline double drelu(double y) {
  //y is not x but relu(x)
  return (y > 0) ? 1. : 0.;
}
__device__ inline double iden(double x) {
  return x;
}
__device__ inline double diden(double y) {
  return 1.;
}
__device__ inline double* softmax(int leng, const double* x, double* y) {
  //y needs to be pre-allocated
  double s = 0.;
  for (int i = 0; i < leng; ++i) {
    s += exp(x[i]);
  }
  for (int i = 0; i < leng; ++i) {
    y[i] = x[i] / s;
  }
  return y;
}
__device__ inline double sseLoss(int L, const double* yhat, const double* yobs) {
  double loss = 0.;
  for (int i = 0; i < L; ++i) {
    loss += (yhat[i] - yobs[i]) * (yhat[i] - yobs[i]) / 2;
  }
  return loss;
}
__device__ inline double* dSseLoss(int L, double* dloss, const double* yhat, const double* yobs) {
  for (int i = 0; i < L; ++i) {
    dloss[i] = (yhat[i] - yobs[i]);
  }
  return dloss;
}
__device__ inline double xenLoss(int leng, const double* yhat, const double* yobs) {
  double loss = 0.;
  for (int i = 0; i < leng; ++i) {
    loss += -yobs[i] * log(yhat[i]);
  }
  return loss;
}
__device__ inline double* dXenLoss(int leng, double* dloss, const double* yhat, const double* yobs) {
  for (int i = 0; i < leng; ++i) {
    dloss[i] = (yhat[i] - yobs[i]);
  }
  return dloss;
}


__device__ void getTransFn(int idTransFn, double(*&fn)(double), double(*&dfn)(double)) {
  switch (idTransFn) {
  case 0:
    fn = logit;
    dfn = dlogit;
    break;
  case 1:
    fn = tanh;
    dfn = dtanh;
    break;
  case 2:
    fn = relu;
    dfn = drelu;
    break;
  case 3:
    fn = iden;
    dfn = diden;
    break;
  default:
    fn = logit;
    dfn = dlogit;
  }
}
__device__ void getLossFn(int idLossFn, double(*&fn)(int, const double*, const double*), double*(*&dfn)(int, double*, const double*, const double*)) {
  switch (idLossFn) {
  case 0:
    fn = sseLoss;
    break;
  case 1:
    fn = xenLoss;
    break;
  }
}
