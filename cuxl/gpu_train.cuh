#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

void setLaunchParams(dim3& tpb, dim3& nb, size_t totThd) {
  tpb.x = (totThd > 128 * 128) ? 256 : 64;
  size_t b = totThd / tpb.x + 1;
  if (b > 256) b = 256; //256 = 128 * 128 / 64, consistent with the top line
  nb.x = (unsigned int)b;
}

__host__ __device__
unsigned int twang6(unsigned int a) {
  //http://burtleburtle.net/bob/hash/integer.html
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}
__host__ __device__
double runif(unsigned int a) {
  return static_cast<double>(twang6(a)) / UINT_MAX;
}