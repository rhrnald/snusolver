#include <cstdio>
#include <cstdlib>

#include "read_matrix.h"

void coo_to_csr(csr_matrix *csr, int *I, int *J, double *val, int nz, int M,
                int N) {
  int i;

  csr->n = M;
  csr->m = N;
  csr->nnz = nz;
  csr->rowptr = (int *)calloc((M + 1), sizeof(int)); // +1 for the last index
  csr->colidx = (int *)malloc(nz * sizeof(int));
  csr->data = (double *)malloc(nz * sizeof(double));

  // Count the number of non-zero entries in each row
  for (i = 0; i < nz; i++) {
    csr->rowptr[I[i] + 1]++;
  }

  // Cumulative sum to get rowptr
  for (i = 0; i < M; i++) {
    csr->rowptr[i + 1] += csr->rowptr[i];
  }

  // Fill in the colidx and data
  for (i = 0; i < nz; i++) {
    int row = I[i];
    int dest = csr->rowptr[row];

    csr->colidx[dest] = J[i];
    csr->data[dest] = val[i];
    csr->rowptr[row]++;
  }

  // Shift rowptr to the right and insert zero at the beginning
  for (i = M; i > 0; i--) {
    csr->rowptr[i] = csr->rowptr[i - 1];
  }
  csr->rowptr[0] = 0;
}

void free_csr_matrix(csr_matrix *csr) {
  free(csr->rowptr);
  free(csr->colidx);
  free(csr->data);
}

csr_matrix read_matrix(int argc, char *argv[]) {
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  int M, N, nz;
  int *I, *J;
  double *val;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else {
    if ((f = fopen(argv[1], "r")) == NULL)
      exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */
  if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
      mm_is_sparse(matcode)) {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  /* find out size of sparse matrix .... */
  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
    exit(1);

  /* reserve memory for matrices */
  I = (int *)malloc(nz * sizeof(int));
  J = (int *)malloc(nz * sizeof(int));
  val = (double *)malloc(nz * sizeof(double));

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
  for (int i = 0; i < nz; i++) {
    fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
    I[i]--; /* adjust from 1-based to 0-based */
    J[i]--;
  }

  if (f != stdin)
    fclose(f);

  /* Convert COO to CSR format */
  csr_matrix csr;

  coo_to_csr(&csr, I, J, val, nz, M, N);

  return csr;
}