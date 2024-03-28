#include "mpi.h"
#include "read_matrix.h"
#include "snusolver.h"

static int np, iam;
static int *sizes;
static int *order;

static MPI_Comm comm = MPI_COMM_WORLD;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &iam);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  int colidx[65] = {0,  2,  7,  12, 14, 16, 1,  2,  2,  5, 9,  12, 3,
                    4,  6,  5,  14, 15, 16, 6,  11, 19, 7, 2,  8,  9,
                    19, 0,  10, 16, 2,  7,  11, 12, 2,  5, 8,  10, 12,
                    13, 4,  10, 13, 3,  8,  9,  14, 19, 8, 12, 15, 19,
                    0,  13, 16, 6,  12, 17, 18, 13, 18, 6, 13, 14, 19};
  int rowptr[21] = {0,  6,  8,  12, 13, 15, 19, 22, 23, 25, 27,
                    30, 34, 40, 43, 48, 52, 55, 59, 61, 65};
  double data[65] = {
      0.24962927, 0.98936737, 0.5596584,  0.82286215, 0.12571838, 0.82745912,
      0.36021587, 0.77324984, 0.45571262, 0.15448031, 0.31913136, 0.90086411,
      0.68234411, 0.6715551,  0.5368658,  0.86844591, 0.5285563,  0.08225238,
      0.73513687, 0.23014697, 0.34630477, 0.52726942, 0.71632364, 0.64321548,
      0.70242432, 0.99793836, 0.0481582,  0.5601762,  0.13580991, 0.25861463,
      0.12299077, 0.11424417, 0.99305946, 0.34917809, 0.44343391, 0.76268574,
      0.50708106, 0.74349204, 0.60026863, 0.95304292, 0.14231242, 0.02761968,
      0.38292221, 0.05910374, 0.36557331, 0.50997393, 0.59215191, 0.15391965,
      0.3834341,  0.37336192, 0.32945818, 0.67772204, 0.24349124, 0.87698404,
      0.2876822,  0.19973612, 0.51757488, 0.6086118,  0.93567469, 0.04185929,
      0.71978582, 0.32142112, 0.40067383, 0.87279979, 0.23431602};

  // csr_matrix A_csr = {n, n, nnz, rowptr, colidx, data};
  csr_matrix A_csr;
  double b[20] = {0.165376, 0.002928, 0.964413, 0.799829, 0.910767,
                  0.41243,  0.310289, 0.939148, 0.109488, 0.265184,
                  0.472562, 0.64517,  0.359738, 0.329123, 0.146323,
                  0.605868, 0.199445, 0.093073, 0.833916, 0.792036};
  double x[20];
  if (!iam) {
    A_csr = read_matrix(argc, argv);
    // b = (double*)malloc(sizeof(double)*A_csr.n);
  }

  int n, nnz;
  if (!iam) {
    n = A_csr.n, nnz = A_csr.nnz;
  }

  MPI_Bcast(&n, sizeof(int), MPI_BYTE, 0, comm);
  MPI_Bcast(&nnz, sizeof(int), MPI_BYTE, 0, comm);
  A_csr.n = n, A_csr.nnz = nnz;

  sizes = (int *)malloc(sizeof(int) * (np * 2 - 1));
  order = (int *)malloc(sizeof(int) * n);

  call_parmetis(A_csr, sizes, order);

  construct_all(A_csr, sizes, order, b);
  distribute_all();
  solve(x);

  free(sizes);
  free(order);
  MPI_Finalize();
}
