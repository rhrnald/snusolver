#ifndef SNUSOLVER_READ_MATRIX
#define SNUSOLVER_READ_MATRIX

#include "mmio.h"
#include "snusolver.h"

void coo_to_csr(csr_matrix *csr, int *I, int *J, double *val, int nz, int M,
                int N);
void free_csr_matrix(csr_matrix *csr);
csr_matrix read_matrix(int argc, char *argv[]);
#endif