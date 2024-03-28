#include "kernel.h"

#include "mkl.h"

void snusolver_LU(dense_matrix &A) {
  // printf("LU %d %d\n", i,j);
  lapack_int n = A.n, m = A.m;

  if (!n || !m)
    return;
  LAPACKE_mkl_dgetrfnp(LAPACK_COL_MAJOR, n, m, A.data, n);
}
void snusolver_trsm_Lxb(dense_matrix &L, dense_matrix &b) {
  MKL_INT n = b.n, m = b.m;
  if (!n || !m)
    return;
  // Lx=b
  // L=n*n, x=n*m, b=n*m;
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, n,
              m, 1, L.data, n, b.data, n);
}
void snusolver_trsm_xUb(dense_matrix &U, dense_matrix &b) {
  // xU=b;
  // x=n*m, U=m*m, b=n*m;
  MKL_INT n = b.n, m = b.m;
  if (!n || !m)
    return;

  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
              n, m, 1, U.data, m, b.data, n);
}

void snusolver_trsm_Uxb(dense_matrix &U, dense_matrix &b) {
  // Ux=b;
  // U=n*n, x=n*m, , b=n*m;
  MKL_INT n = b.n, m = b.m;
  if (!n || !m)
    return;

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
              n, m, 1, U.data, n, b.data, n);
}
void snusolver_gemm(dense_matrix &A, dense_matrix &B, dense_matrix &C) {
  // A=m*k, B=k*n, C=m*n;

  MKL_INT m = A.n; //=C.n;
  MKL_INT k = A.m; //=B.n;
  MKL_INT n = B.m; //=C.m;
  if (!m || !n || !k)
    return;

  // if(iam==0) {start_time = std::chrono::system_clock::now();}

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, -1, A.data, m,
              B.data, k, 1, C.data, m);
  // if(iam==0) {END("gemm (" << m << " " << n << " " << k << ") ")}
}