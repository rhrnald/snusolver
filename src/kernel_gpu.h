#ifndef SNUSOLVER_GPU_KERNEL
#define SNUSOLVER_GPU_KERNEL
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>

#include "snusolver.h"

static int iam;
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "%d: GPUassert: %s %s %d\n", iam, cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

void snusolver_LU_gpu(dense_matrix &A, cusolverDnHandle_t handle);
void snusolver_trsm_Lxb_gpu(dense_matrix &L, dense_matrix &b,
                            cublasHandle_t handle);
void snusolver_trsm_xUb_gpu(dense_matrix &U, dense_matrix &b,
                            cublasHandle_t handle);
void snusolver_trsm_Uxb_gpu(dense_matrix &U, dense_matrix &b,
                            cublasHandle_t handle);
void snusolver_gemm_gpu(dense_matrix &A, dense_matrix &B, dense_matrix &C,
                        cublasHandle_t handle);
#endif