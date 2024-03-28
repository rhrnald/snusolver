#include <stdio.h>

#include <chrono>
#include <iostream>

#include "kernel_gpu.h"
static std::chrono::time_point<std::chrono::system_clock> start_time, end_time;

static double *Workspace;
static int Lwork_size = 0;
static const double alpha = 1.0;

void snusolver_LU_gpu(dense_matrix &A, cusolverDnHandle_t cusolverHandle) {
  int Lwork;
  int n = A.n, m = A.m;
  if (!n || !m)
    return;
  cusolverDnDgetrf_bufferSize(cusolverHandle, n, m, A.data_gpu, m, &Lwork);
  if (Lwork > Lwork_size) {
    if (!Lwork_size)
      gpuErrchk(cudaFree(Workspace));
    gpuErrchk(cudaMalloc((void **)&Workspace, Lwork * sizeof(double)));
    Lwork_size = Lwork;
  }
  cusolverDnDgetrf(cusolverHandle, n, m, A.data_gpu, m, Workspace, nullptr,
                   nullptr);
}
void snusolver_trsm_Lxb_gpu(dense_matrix &L, dense_matrix &b,
                            cublasHandle_t handle) {
  int n = b.n, m = b.m;
  if (!n || !m)
    return;

  // cblas_dtrsm (CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
  // CblasUnit,n,m,1,L.data,n,b.data,m);
  cublasSideMode_t side = CUBLAS_SIDE_LEFT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t trans = CUBLAS_OP_N;
  cublasDiagType_t diag = CUBLAS_DIAG_UNIT;

  cublasDtrsm(handle, side, uplo, trans, diag, n, m, &alpha, L.data_gpu, n,
              b.data_gpu, n);
}
void snusolver_trsm_xUb_gpu(dense_matrix &U, dense_matrix &b,
                            cublasHandle_t handle) {
  int n = b.n, m = b.m;
  if (!n || !m)
    return;
  // cblas_dtrsm (CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans,
  // CblasNonUnit,n,m,1,U.data,m,b.data,m);

  cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  cublasOperation_t trans = CUBLAS_OP_N;
  cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

  cublasDtrsm(handle, side, uplo, trans, diag, n, m, &alpha, U.data_gpu, m,
              b.data_gpu, n);
  // cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
  // CUBLAS_DIAG_NON_UNIT,n,m,&alpha, U.data,m,b.data,n);
}

void snusolver_trsm_Uxb_gpu(dense_matrix &U, dense_matrix &b,
                            cublasHandle_t handle) {
  int n = b.n, m = b.m;
  if (!n || !m)
    return;
  // cblas_dtrsm (CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
  // CblasNonUnit,n,m,1,U.data,n,b.data,m);
  cublasSideMode_t side = CUBLAS_SIDE_LEFT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  cublasOperation_t trans = CUBLAS_OP_N;
  cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

  cublasDtrsm(handle, side, uplo, trans, diag, n, m, &alpha, U.data_gpu, n,
              b.data_gpu, n);
  // cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
  // CUBLAS_DIAG_NON_UNIT,n,m,&alpha, U.data,n,b.data,n);
}

void snusolver_gemm_gpu(dense_matrix &A, dense_matrix &B, dense_matrix &C,
                        cublasHandle_t handle) {
  int m = A.n, k = A.m, n = B.m;
  // A=m*k, B=k*n, C=m*n
  if (!m || !n || !k)
    return;
  // cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, m,n,k, -1, A.data,
  // k, B.data, n, 1, C.data, n);
  /*cublasStatus_t cublasDgemm(cublasHandle_t handle,
                         cublasOperation_t transa, cublasOperation_t transb,
                         int m, int n, int k,
                         const double          *alpha,
                         const double          *A, int lda,
                         const double          *B, int ldb,
                         const double          *beta,
                         double          *C, int ldc)*/
  cublasOperation_t trans = CUBLAS_OP_N;
  const double alpha = -1.0;
  const double beta = 1.0;
  cublasDgemm(handle, trans, trans, m, n, k, &alpha, A.data_gpu, m, B.data_gpu,
              k, &beta, C.data_gpu, m);
}