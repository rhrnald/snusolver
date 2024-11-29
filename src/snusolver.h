#ifndef SNUSOLVER
#define SNUSOLVER
#include "SnuMat.h"

void initialize();
void solve(csr_matrix A_csr, double *b, double *x);
#endif