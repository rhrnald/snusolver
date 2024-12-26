#include <stdio.h>

#include <chrono>
#include <iostream>
#include <vector>
#include <tuple>
#include <mpi.h>
#include <fstream>
#include <ctime>     // Include for time functions
#include <cstdio>
#include <cusolverDn.h>


#include "param.h"
#include "matrix.h"

static int iam;
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "%d: GPUassert: %s %s %d\n", iam, cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

static std::chrono::time_point<std::chrono::system_clock> s,e;
#define START() s = std::chrono::system_clock::now();
#define END() e = std::chrono::system_clock::now();
#define GET()                                                                  \
  (std::chrono::duration_cast<std::chrono::duration<double>>(                   \
       (e = std::chrono::system_clock::now()) - s)                             \
       .count())

using namespace std;
static double *Workspace;
static int Lwork_size = 0;
static const double alpha = 1.0;
static int *status;
static vector<tuple<long long,double>> v_getrf, v_trsm, v_gemm;

void snusolver_LU_gpu(dense_matrix &A, cusolverDnHandle_t cusolverHandle) {
  int Lwork;
  int n = A.n, m = A.m;
  if (!n || !m)
    return;
  cusolverDnDgetrf_bufferSize(cusolverHandle, n, m, A.data_gpu, m, &Lwork);
  if (Lwork > Lwork_size) {
    if (Lwork_size) {
      gpuErrchk(cudaFree(Workspace));
    }
    else {
      gpuErrchk(cudaMalloc((void **)&status, sizeof(int)));
    }

    gpuErrchk(cudaMalloc((void **)&Workspace, Lwork * sizeof(double)));
    Lwork_size = Lwork;
  }

#ifdef MEASURE_FLOPS
  START();
  cudaDeviceSynchronize();
#endif
  cusolverDnDgetrf(cusolverHandle, n, m, A.data_gpu, m, Workspace, nullptr,
                   status);
#ifdef MEASURE_FLOPS
  double time = GET();
  v_getrf.push_back({1ll*n*n*n/3,time});
#endif
}
void snusolver_trsm_Lxb_gpu(dense_matrix &L, dense_matrix &b,
                            cublasHandle_t handle) {
  int n = b.n, m = b.m;
  if (!n || !m)
    return;

  // cblas_dtrsm (CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
  // CblasUnit,n,m,1,L.data,n,b.data,m);
  cublasSideMode_t side = CUBLAS_SIDE_LEFT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t trans = CUBLAS_OP_N;
  cublasDiagType_t diag = CUBLAS_DIAG_UNIT;

#ifdef MEASURE_FLOPS
  cudaDeviceSynchronize();
  START();
#endif
  cublasDtrsm(handle, side, uplo, trans, diag, n, m, &alpha, L.data_gpu, n,
              b.data_gpu, n);
#ifdef MEASURE_FLOPS
  double time = GET();
  v_trsm.push_back({1ll*n*n*m/2,time});
#endif
}
void snusolver_trsm_xUb_gpu(dense_matrix &U, dense_matrix &b,
                            cublasHandle_t handle) {
  int n = b.n, m = b.m;
  if (!n || !m)
    return;
  // cblas_dtrsm (CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans,
  // CblasNonUnit,n,m,1,U.data,m,b.data,m);

  cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  cublasOperation_t trans = CUBLAS_OP_N;
  cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

#ifdef MEASURE_FLOPS
  cudaDeviceSynchronize();
  START();
#endif
  cublasDtrsm(handle, side, uplo, trans, diag, n, m, &alpha, U.data_gpu, m,
              b.data_gpu, n);
#ifdef MEASURE_FLOPS
  double time = GET();
  v_trsm.push_back({1ll*n*m*m/2,time});
#endif
}

void snusolver_trsm_Uxb_gpu(dense_matrix &U, dense_matrix &b,
                            cublasHandle_t handle) {
  int n = b.n, m = b.m;
  if (!n || !m)
    return;
  // cblas_dtrsm (CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
  // CblasNonUnit,n,m,1,U.data,n,b.data,m);
  cublasSideMode_t side = CUBLAS_SIDE_LEFT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  cublasOperation_t trans = CUBLAS_OP_N;
  cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

#ifdef MEASURE_FLOPS
  cudaDeviceSynchronize();
  START();
#endif
  cublasDtrsm(handle, side, uplo, trans, diag, n, m, &alpha, U.data_gpu, n,
              b.data_gpu, n);
#ifdef MEASURE_FLOPS
  double time = GET();
  v_trsm.push_back({1ll*n*n*m/2,time});
#endif
}

void snusolver_gemm_gpu(dense_matrix &A, dense_matrix &B, dense_matrix &C,
                        cublasHandle_t handle) {
  int m = A.n, k = A.m, n = B.m;
  // A=m*k, B=k*n, C=m*n
  if (!m || !n || !k)
    return;
  // cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, m,n,k, -1, A.data,
  // k, B.data, n, 1, C.data, n);
  /*cublasStatus_t cublasDgemm(cublasHandle_t handle,
                         cublasOperation_t transa, cublasOperation_t transb,
                         int m, int n, int k,
                         const double          *alpha,
                         const double          *A, int lda,
                         const double          *B, int ldb,
                         const double          *beta,
                         double          *C, int ldc)*/
  cublasOperation_t trans = CUBLAS_OP_N;
  const double alpha = -1.0;
  const double beta = 1.0;
#ifdef MEASURE_FLOPS
  cudaDeviceSynchronize();
  START();
#endif
  cublasDgemm(handle, trans, trans, m, n, k, &alpha, A.data_gpu, m, B.data_gpu,
              k, &beta, C.data_gpu, m);
#ifdef MEASURE_FLOPS
  double time = GET();
  v_gemm.push_back({1ll*n*m*k*2,time});
#endif
}

void flattenData(const vector<tuple<long long, double>>& tuples, vector<double>& flatData) {
    for (const auto& t : tuples) {
        flatData.push_back(static_cast<double>(get<0>(t))); // int -> double
        flatData.push_back(get<1>(t)); // double 그대로
    }
}

// Function to gather varying data sizes and write to a file in the master process

void log_gpu_flop() { 
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Local data for each process (this can be of varying sizes)
    // Flatten local data into a simple array for each vector

    // Handle GETRF, TRSM, and GEMM vectors
    vector<vector<tuple<long long,double>>> vectors = {v_getrf, v_trsm, v_gemm};

    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char filename[100];
    snprintf(filename, sizeof(filename), "log_cuda_%04d-%02d-%02d_%02d-%02d-%02d.txt",
              t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
              t->tm_hour, t->tm_min, t->tm_sec);
    ofstream file;
    if(!rank) {
      // Open the file with the generated filename
      file.open(filename);
      if (!file.is_open()) {
          cerr << "Unable to open file: " << filename << endl;
          return;
      }
    }
    // Gather and process each vector
    for (int vecIndex = 0; vecIndex < vectors.size(); ++vecIndex) {
        vector<double> flatLocalData;
        flattenData(vectors[vecIndex], flatLocalData);

        // Size of local data
        int localCount = flatLocalData.size();

        // Gather the counts of data from each process
        vector<int> recvCounts(size);
        MPI_Gather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, comm);

        // Calculate displacements for receiving data in the master node
        vector<int> displs(size, 0);
        if (rank == 0) {
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i - 1] + recvCounts[i - 1];
            }
        }

        // Total size of gathered data on the master node
        int totalCount = 0;
        if (rank == 0) {
            totalCount = displs[size - 1] + recvCounts[size - 1];
        }

        // Prepare receive buffer on the master node
        vector<double> gatheredData;
        if (rank == 0) {
            gatheredData.resize(totalCount);
        }

        // Gather varying amounts of data from each process to the master node
        MPI_Gatherv(flatLocalData.data(), localCount, MPI_DOUBLE,
                    gatheredData.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                    0, comm);

        // Master node writes gathered data to a file
        if (rank == 0) {
            // Generate the filename with timestamp
            

            // Print label for each vector (GETRF, TRSM, GEMM)
            if (vecIndex == 0) file << "GETRF" << endl;
            else if (vecIndex == 1) file << "TRSM" << endl;
            else if (vecIndex == 2) file << "GEMM" << endl;

            // Write data in chunks based on original process
            for (size_t i = 0; i < gatheredData.size(); i += 2) {
                if (i + 3 < gatheredData.size()) {
                    file << " " << static_cast<long long>(gatheredData[i]) << " "
                         << gatheredData[i + 1] << endl; // Write each entry on a new line
                }
            }

        }
    }
  if(!rank) {
    file.close();
    cout << "Data written to file: " << filename << endl;
  }
}