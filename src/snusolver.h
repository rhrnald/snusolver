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
} csr_matrix;

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

void call_parmetis(csr_matrix A, int *sizes, int *order);
void construct_all(csr_matrix A_csr, int *sizes, int *order, double *b);
void distribute_all();
void factsolve(double *b_ret);
void initialize();

void solve(csr_matrix A_csr, double *b, double *x);