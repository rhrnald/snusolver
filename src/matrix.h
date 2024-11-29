#ifndef SNUSOLVER_MATRIX
#define SNUSOLVER_MATRIX
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

  // void dense_matrix::toCPU() {
  //   cudaMemcpy(data, data_gpu, n * m * sizeof(double), cudaMemcpyDeviceToHost);
  // }
  // void dense_matrix::toGPU() {
  //   // if (able)
  //   cudaMemcpy(data_gpu, data, n * m * sizeof(double), cudaMemcpyHostToDevice);
  // }
};
#endif