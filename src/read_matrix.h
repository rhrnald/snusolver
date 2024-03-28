#pragma once

#include "snusolver.h"
#include "mmio.h"

void coo_to_csr(csr_matrix *csr, int *I, int *J, double *val, int nz, int M, int N);
void free_csr_matrix(csr_matrix *csr);
csr_matrix read_matrix(int argc, char *argv[]);
