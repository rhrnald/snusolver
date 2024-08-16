#include "mpi.h"
#include "read_matrix.h"
#include "snusolver.h"
#include <random>

static int np, iam;

static MPI_Comm comm = MPI_COMM_WORLD;

#include <chrono>

static std::chrono::time_point<std::chrono::system_clock> sss, eee;

void check(csr_matrix A, double *b, double *x) {
  int n = A.n;
  double M = 0;
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int ptr = A.rowptr[i]; ptr < A.rowptr[i + 1]; ptr++) {
      int c = A.colidx[ptr];
      sum += A.data[ptr] * x[c];
    }
    sum-=b[i];
    if (M < sum)
      M = sum;
    if (M < -sum)
      M = -sum;
  }
  printf("max |Ax-b| = %e\n", M);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &iam);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  createHandle();

  csr_matrix A_csr;
  double *b, *x;
  if (!iam) {
    std::cout << "Reading matrix" << std::endl;
    A_csr = read_matrix(argc, argv);
    std::cout << "Read matrix done!" << std::endl;

    b = (double *)malloc(sizeof(double) * A_csr.n);
    x = (double *)malloc(sizeof(double) * A_csr.n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < A_csr.n; i++) {
      b[i] = dis(gen);
    }
  }

  solve(A_csr, b, x);

  if (!iam) {
    check(A_csr, b, x);
  }

  MPI_Finalize();
}
