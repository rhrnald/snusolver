#include <memory>
#include "mpi.h"

#include "SnuMat.h"
#include <cstdio>


#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n",cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

void call_parmetis(csr_matrix A, int *sizes, int *order);
void SnuMat::gather_data_b() {
  for (int i = num_block; i >= 1; i--) {
    if (!(who[i])) {
      memcpy(perm_b + old_block_start[i], b[i].data,
             block_size[i] * sizeof(double));
    } else {
      MPI_Recv(perm_b + old_block_start[i], block_size[i], MPI_DOUBLE, who[i],
              0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

void SnuMat::return_data_b() {
  for (auto &i : my_block) {
    MPI_Send(b[i].data, b[i].n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
}
void SnuMat::b_togpu() {
  gpuErrchk(cudaMemcpy(this->_b_gpu, this->_b, this->local_b_rows * sizeof(double),
                       cudaMemcpyHostToDevice));
}
void SnuMat::b_tocpu() {
  gpuErrchk(cudaMemcpy(this->_b, this->_b_gpu, this->local_b_rows * sizeof(double),
                       cudaMemcpyDeviceToHost));
}

SnuMat::SnuMat(csr_matrix A_csr, double *b, cublasHandle_t handle, cusolverDnHandle_t cusolverHandle) {
  this->handle=handle;
  this->cusolverHandle=cusolverHandle;

  MPI_Comm_size(MPI_COMM_WORLD, &(this->np));
  MPI_Comm_rank(MPI_COMM_WORLD, &(this->iam));
  num_block=np+np-1;

  if (!iam) {
    n = A_csr.n, nnz = A_csr.nnz;
  }
  MPI_Bcast(&n, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nnz, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  A_csr.n = n, A_csr.nnz = nnz;

  int* sizes = (int *)malloc(sizeof(int) * (np * 2 - 1));
  order = (int *)malloc(sizeof(int) * n);
  
  // construct_all(A_csr, sizes, order, b);
  call_parmetis(A_csr, sizes, order);
  make_who();
  construct_all(A_csr, sizes, order, b);
  distribute_matrix();
}

void dense_matrix::toCPU() {
  cudaMemcpy(data, data_gpu, n * m * sizeof(double), cudaMemcpyDeviceToHost);
}
void dense_matrix::toGPU() {
  // if (able)
  cudaMemcpy(data_gpu, data, n * m * sizeof(double), cudaMemcpyHostToDevice);
}