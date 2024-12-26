#ifndef SNUSOLVER_KERNEL_CUDA
#define SNUSOLVER_KERNEL_CUDA
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "SnuMat.h"

#include <stdio.h>


void snusolver_LU_gpu(dense_matrix &A, cusolverDnHandle_t handle);
void snusolver_trsm_Lxb_gpu(dense_matrix &L, dense_matrix &b,
                            cublasHandle_t handle);
void snusolver_trsm_xUb_gpu(dense_matrix &U, dense_matrix &b,
                            cublasHandle_t handle);
void snusolver_trsm_Uxb_gpu(dense_matrix &U, dense_matrix &b,
                            cublasHandle_t handle);
void snusolver_gemm_gpu(dense_matrix &A, dense_matrix &B, dense_matrix &C,
                        cublasHandle_t handle);
void log_gpu_flop();
void log_sparse_flop();
void log_mkl_flop();
#endif