#include <algorithm>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>

#include <chrono>
#include <map>
#include <iostream>

#include "kernel.h"
#include "mpi.h"
#include "snusolver.h"

#include "SnuMat.h"

using namespace std;

static int np,iam;
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

static cublasHandle_t handle;
static cusolverDnHandle_t cusolverHandle;

void initialize() {
  MPI_Comm_rank(MPI_COMM_WORLD, &iam);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  char *local_size_env = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  int nproc_pernode = local_size_env ? atoi(local_size_env) : -1;
  int ngpu = 4;
  gpuErrchk(cudaGetDeviceCount(&ngpu));


  gpuErrchk(cudaSetDevice((iam % nproc_pernode) / (nproc_pernode / ngpu)));  //nproc_pernode must be multiple of ngpu

  cublasCreate(&handle);
  cusolverDnCreate(&cusolverHandle);
}

void solve(csr_matrix A_csr, double *b, double *x) {
  MPI_Barrier(MPI_COMM_WORLD);
  if(!iam) TIMER_START("Total start");

  SnuMat Ab(A_csr, b, handle, cusolverHandle);
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(!iam) TIMER_START("Factsolve start");
  Ab.solve(x);
  MPI_Barrier(MPI_COMM_WORLD);
  if(!iam) TIMER_END("Factsolve end");

  MPI_Barrier(MPI_COMM_WORLD);
  if(!iam) TIMER_END("Total end");

  gatherAndWriteData();
}