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



int SnuMat::_who(int block) {
  while (block >= np + np)
    block /= 2;
  while (block < np)
    block = block + block + 1;
  return np + np - block - 1;
}


void SnuMat::make_who() {
  max_level=0;
  for (int i = np; i > 1; i /= 2)
    max_level++;

  vector<int> level(np+np, 0);

  for (int i = 1; i <= num_block; i++)
    who[i] = _who(i);
  for (int i = num_block; i >= 1; i--)
    if (who[i] == iam)
      my_block.push_back(i);
  for (int e = 1; e <= num_block; e++) {
    for (int i = e; i > 1; i /= 2)
      level[e]++;
  }
  for (auto &e : my_block) {
    my_block_level[level[e]].push_back(e);
  }

  vector<int> vst(num_block + 1, 0);
  for (int i = 1; i <= num_block; i++) {
    if (who[i] == iam) {
      for (int j = i; j >= 1; j /= 2)
        vst[j] = 1;
    }
  }

  for (int i = num_block; i >= 1; i--)
    if (vst[i]) {
      all_parents.push_back(i);
    }
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

  sizes = (int *)malloc(sizeof(int) * (np * 2 - 1));
  order = (int *)malloc(sizeof(int) * n);

#ifdef MEASURE_TIME
  MPI_Barrier(MPI_COMM_WORLD);
  if(!iam) TIMER_START("Preporcess start");
#endif


#ifdef MEASURE_TIME
    MPI_Barrier(MPI_COMM_WORLD);
    if(!iam) TIMER_START("Parmetis start");
#endif

    call_parmetis(A_csr, sizes, order);

#ifdef MEASURE_TIME
    MPI_Barrier(MPI_COMM_WORLD);
    if(!iam) TIMER_END("Parmetis done");
#endif

    construct_structure(A_csr, sizes, order, b);
    distribute_structure();
    malloc_matrix();
    distribute_data();

#ifdef MEASURE_TIME
  MPI_Barrier(MPI_COMM_WORLD);
  if(!iam) TIMER_END("Preprocess done");
#endif
}

void dense_matrix::toCPU() {
  cudaMemcpy(data, data_gpu, n * m * sizeof(double), cudaMemcpyDeviceToHost);
}
void dense_matrix::toGPU() {
  // if (able)
  cudaMemcpy(data_gpu, data, n * m * sizeof(double), cudaMemcpyHostToDevice);
}