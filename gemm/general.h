#pragma once

#include "cuda_runtime.h"
#include <iostream>
// 检查是否调用成功
#define CHECK(call)                                          \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess)                                \
    {                                                        \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code: %d, reason: %s\n", error,       \
              cudaGetErrorString(error));                    \
      exit(1);                                               \
    }                                                        \
  }

void initialData(float *ip, int size)
{
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; ++i)
  {
    // ip[i] =(float)(rand() & 0xFF)/10.0f;
    ip[i] = 1.0f;
  }
}

void viewMat(const float *p, const int r, const int c)
{
  for (int i = 0; i < r; ++i)
  {
    for (int j = 0; j < c; ++j)
    {
      std::cout << p[i * c + j] << " ";
    }
    std::cout << std::endl;
  }
}
