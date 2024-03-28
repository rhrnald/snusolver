#pragma once

typedef struct coo_matrix {
  int n, m, nnz;
  int *row, *col;
  double *data;
} coo_matrix;

typedef struct csr_matrix {
  int n, m, nnz;
  int *rowptr, *colidx;
  double *data;
};

typedef struct csc_matrix {
  int n, m, nnz;
  int *rowidx, *colptr;
  double *data;
} csc_matrix;

class dense_matrix {
public:
  int n, m;
  double *data;
  double *data_gpu;
  bool able;

  void toCPU();
  void toGPU();
};

int construct_all(csr_matrix A_csr, int *sizes, int *order);