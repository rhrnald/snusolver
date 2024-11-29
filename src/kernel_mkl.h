#ifndef SNUSOLVER_KERNEL_MKL
#define SNUSOLVER_KERNEL_MKL
#include <stdio.h>

#include "snusolver.h"

void snusolver_LU(dense_matrix &A);
void snusolver_trsm_Lxb(dense_matrix &L, dense_matrix &b);
void snusolver_trsm_xUb(dense_matrix &U, dense_matrix &b);
void snusolver_trsm_Uxb(dense_matrix &U, dense_matrix &b);

void snusolver_LU_sparse(coo_matrix &A, csr_matrix &LU);
void snusolver_trsm_Lxb_sparse(csr_matrix &L, dense_matrix &b, int i, int j);
void snusolver_trsm_xUb_sparse(csr_matrix &U, dense_matrix &b, int i, int j);
void snusolver_trsm_Uxb_sparse(csr_matrix &U, dense_matrix &b, int i, int j);

void snusolver_gemm(dense_matrix &A, dense_matrix &B, dense_matrix &C);

#endif