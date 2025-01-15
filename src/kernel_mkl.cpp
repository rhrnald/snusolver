#include "kernel.h"

#include "mkl.h"
#include <fstream>
#include <chrono>

static std::chrono::time_point<std::chrono::system_clock> s,e;
#define START() s = std::chrono::system_clock::now();
#define END() e = std::chrono::system_clock::now();
#define GET()                                                                  \
  (std::chrono::duration_cast<std::chrono::duration<double>>(                   \
       (e = std::chrono::system_clock::now()) - s)                             \
       .count())
static vector<tuple<long long,double>> v_getrf, v_trsm, v_gemm;

void snusolver_LU(dense_matrix &A) {
  // printf("LU %d %d\n", i,j);
  lapack_int n = A.n, m = A.m;

  if (!n || !m)
    return;
  
#ifdef MEASURE_FLOPS
  START();
#endif
  LAPACKE_mkl_dgetrfnp(LAPACK_COL_MAJOR, n, m, A.data, n);
#ifdef MEASURE_FLOPS
  double time = GET();
  v_getrf.push_back({1ll*n*n*n/3,time});
#endif
}
void snusolver_trsm_Lxb(dense_matrix &L, dense_matrix &b) {
  MKL_INT n = b.n, m = b.m;
  if (!n || !m)
    return;
  // Lx=b
  // L=n*n, x=n*m, b=n*m;
#ifdef MEASURE_FLOPS
  START();
#endif
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, n,
              m, 1, L.data, n, b.data, n);
#ifdef MEASURE_FLOPS
  double time = GET();
  v_trsm.push_back({1ll*n*n*m/2,time});
#endif
}
void snusolver_trsm_xUb(dense_matrix &U, dense_matrix &b) {
  // xU=b;
  // x=n*m, U=m*m, b=n*m;
  MKL_INT n = b.n, m = b.m;
  if (!n || !m)
    return;

#ifdef MEASURE_FLOPS
  START();
#endif
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
              n, m, 1, U.data, m, b.data, n);
#ifdef MEASURE_FLOPS
  double time = GET();
  v_trsm.push_back({1ll*n*m*m*2,time});
#endif
}

void snusolver_trsm_Uxb(dense_matrix &U, dense_matrix &b) {
  // Ux=b;
  // U=n*n, x=n*m, , b=n*m;
  MKL_INT n = b.n, m = b.m;
  if (!n || !m)
    return;

#ifdef MEASURE_FLOPS
  START();
#endif
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
              n, m, 1, U.data, n, b.data, n);
#ifdef MEASURE_FLOPS
  double time = GET();
  v_trsm.push_back({1ll*n*n*m/2,time});
#endif
}
void snusolver_gemm(dense_matrix &A, dense_matrix &B, dense_matrix &C) {
  // A=m*k, B=k*n, C=m*n;

  MKL_INT m = A.n; //=C.n;
  MKL_INT k = A.m; //=B.n;
  MKL_INT n = B.m; //=C.m;
  if (!m || !n || !k)
    return;

  // if(iam==0) {start_time = std::chrono::system_clock::now();}

#ifdef MEASURE_FLOPS
  START();
#endif
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, -1, A.data, m,
              B.data, k, 1, C.data, m);
#ifdef MEASURE_FLOPS
  double time = GET();
  v_gemm.push_back({1ll*n*m*k*2,time});
#endif
}

void flattenData(const vector<tuple<long long, double>>& tuples, vector<double>& flatData);
void log_mkl_flop() { 
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Local data for each process (this can be of varying sizes)
    // Flatten local data into a simple array for each vector

    // Handle GETRF, TRSM, and GEMM vectors
    
    vector<vector<tuple<long long,double>>> vectors = {v_getrf, v_trsm, v_gemm};
    {
      double sum=0;
      for(auto &e: vectors) {
        for(auto &f: e) {
          sum+=get<1>(f);
        }
      }
      // printf("process %d computation time : %lf\n", rank, sum);
    }
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char filename[100];
    snprintf(filename, sizeof(filename), "log_mkl_%04d-%02d-%02d_%02d-%02d-%02d.txt",
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