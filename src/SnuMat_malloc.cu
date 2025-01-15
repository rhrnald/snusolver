#include <algorithm>
#include "mpi.h"

#include "SnuMat.h"

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n",
    cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}


void SnuMat::malloc_LU(int i, int j, int lvl, double *data, double *data_gpu) {
  int row = block_size[i];
  int col = block_size[j];

  static int malloc_LU_bias=0;
  // double *data, *data_gpu = nullptr;
  // data = (double *)malloc(row * col * sizeof(double));
  // gpuErrchk(cudaHostAlloc((void**)&data, row * col * sizeof(double), cudaHostAllocDefault));
  if (lvl <= offlvl && (!(iam & 1))) {
    // gpuErrchk(cudaMalloc((void **)&data_gpu, row * col * sizeof(double)));
    LU[{i, j}] = {row, col, data+malloc_LU_bias, data_gpu+malloc_LU_bias, 1};
  } else {
    LU[{i, j}] = {row, col, data+malloc_LU_bias, data_gpu+malloc_LU_bias, 0};
  }  
  malloc_LU_bias+=row*col;
}
void SnuMat::free_LU(int i, int j, int lvl) {
  free(LU[{i, j}].data);
  if (lvl <= offlvl && (!(iam & 1)))
    cudaFree(LU[{i, j}].data_gpu);
}
void SnuMat::clear_LU(int i, int j) {
  auto &e = LU[{i, j}];
  int row = e.n;
  int col = e.m;

  std::fill(e.data, e.data + row * col, 0.0);
  if (e.able)
    cudaMemset(e.data_gpu, 0, row * col * sizeof(double));
}

__global__ void copySparseToDense(const int *row, const int *col,
                                  const double *data, double *dmat, int r,
                                  int nnz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nnz) {
    *(dmat + (*(col + idx) * r + *(row + idx))) += *(data + idx);
  }
}

void SnuMat::malloc_all_LU() {
  
  double *data, *data_gpu = nullptr;
  if (offlvl>=0 && (!(iam & 1))) {
    gpuErrchk(cudaMalloc((void **)&data_gpu, dense_row*dense_row * sizeof(double)));
    gpuErrchk(cudaHostAlloc((void**)&data, dense_row*dense_row * sizeof(double), cudaHostAllocDefault));
  } else {
    data = (double *)malloc(dense_row*dense_row * sizeof(double));
  }

  for (auto &i : (all_parents)) {
    if (i >= np)
      continue;
    int lvl = level[i];
    malloc_LU(i, i, lvl, data, data_gpu);
    for (int j = i / 2; j >= 1; j /= 2) {
      malloc_LU(i, j, lvl, data, data_gpu);
      malloc_LU(j, i, lvl, data, data_gpu);
    }
  }
}
void SnuMat::free_all_LU() {
  for (auto &i : (all_parents)) {
    if (i >= np)
      continue;
    int lvl = level[i];
    free_LU(i, i, lvl);
    for (int j = i / 2; j >= 1; j /= 2) {
      free_LU(i, j, lvl);
      free_LU(j, i, lvl);
    }
  }
}

void SnuMat::malloc_all_b() {
  int sum = 0;
  for (auto &i : (all_parents))
    sum += block_size[i];
  local_b_rows = sum;

  // _b = (double *)malloc(sizeof(double) * sum);
  // gpuErrchk(cudaHostAlloc((void**)&_b, sum * sizeof(double), cudaHostAllocDefault));

  if (offlvl >= 0 && (!(iam & 1))) {
    gpuErrchk(cudaMalloc((void **)&_b_gpu, sum * sizeof(double)));
    gpuErrchk(cudaHostAlloc((void**)&_b, sum * sizeof(double), cudaHostAllocDefault));
  } else {
    _b = (double *)malloc(sizeof(double) * sum);
  }

  sum = 0;
  for (auto &i : (all_parents)) {
    int sz = block_size[i];
    b[i] = {sz, 1, _b + sum, _b_gpu + sum, false};
    sum += sz;
  }
}
void SnuMat::free_all_b() {
  free(_b);
  cudaFree(_b_gpu);
}

void SnuMat::clear_all_LU() {
  for (auto &i : (all_parents)) {
    if (i >= np)
      continue;
    clear_LU(i, i);
    for (int j = i / 2; j >= 1; j /= 2) {
      clear_LU(i, j);
      clear_LU(j, i);
    }
  }

  
  std::fill(MM_buf, MM_buf + dense_row * dense_row, 0);
  if ((!(iam & 1)))
    cudaMemset(MM_buf_gpu, 0, dense_row * dense_row * sizeof(double));
}

void SnuMat::malloc_matrix() {
  malloc_all_LU();
  malloc_all_b();
  core_preprocess();
  clear_all_LU();


  if (offlvl >= 0 && (!(iam & 1))) {
    max_nnz = max(max_nnz, n);
    // gpuErrchk(cudaMalloc((void **)&gpu_row_buf, max_nnz * sizeof(int)));
    // gpuErrchk(cudaMalloc((void **)&gpu_col_buf, max_nnz * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&gpu_data_buf, max_nnz * sizeof(double)));
  }
}