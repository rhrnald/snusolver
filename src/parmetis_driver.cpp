#include <mpi.h>

#include <cstdio>
#include <map>
#include <vector>

#include "parmetis.h"
#include "snusolver.h"

using namespace std;

static int np, iam;
static MPI_Comm comm = MPI_COMM_WORLD;

static csr_matrix addTransposeEliminateDiag(csr_matrix A, int *buf1,
                                            int *buf2) {
  int rows = A.n, cols = A.m, nnz = A.nnz, *rowidx = A.colidx,
      *colptr = A.rowptr;
  vector<int> rowidx_t(nnz);
  vector<int> colptr_t(cols + 1, 0);

  for (int i = 0; i < nnz; i++)
    colptr_t[rowidx[i]]++;
  for (int i = 0; i < cols; i++)
    colptr_t[i + 1] += colptr_t[i];

  for (int c = cols - 1; c >= 0; c--) {
    for (int i = colptr[c]; i < colptr[c + 1]; i++) {
      int r = rowidx[i];
      colptr_t[r]--;
      rowidx_t[colptr_t[r]] = c;
    }
  }

  int new_nnz = 0;
  int *new_colptr = buf1;
  int *new_rowidx = buf2;
  for (int c = 0; c < cols; c++) {
    new_colptr[c] = new_nnz;
    int l1 = colptr[c], r1 = colptr[c + 1], l2 = colptr_t[c],
        r2 = colptr_t[c + 1];
    while (l1 < r1 && l2 < r2) {
      if (rowidx[l1] < rowidx_t[l2]) {
        if (rowidx[l1] != c)
          new_rowidx[new_nnz++] = rowidx[l1];
        l1++;
      } else if (rowidx[l1] > rowidx_t[l2]) {
        if (rowidx_t[l2] != c)
          new_rowidx[new_nnz++] = rowidx_t[l2];
        l2++;
      } else {
        if (rowidx[l1] != c)
          new_rowidx[new_nnz++] = rowidx[l1];
        l1++, l2++;
      }
    }
    while (l1 < r1) {
      if (rowidx[l1] != c)
        new_rowidx[new_nnz++] = rowidx[l1];
      l1++;
    }
    while (l2 < r2) {
      if (rowidx_t[l2] != c)
        new_rowidx[new_nnz++] = rowidx_t[l2];
      l2++;
    }
  }
  new_colptr[cols] = new_nnz;

  return {rows, cols, new_nnz, new_colptr, new_rowidx, nullptr};
}

// int main_parmetis(int argc, char *argv[]) {
void call_parmetis(csr_matrix A, int *sizes, int *order) {
  MPI_Comm_rank(comm, &iam);
  MPI_Comm_size(comm, &np);

  int loc_n, loc_nnz;
  int numflag = 0;
  int options[4] = {0, 0, 0, 1};
  int *vtxdist, *colidx_loc, *rowptr_loc;
  int nnz, n = A.n;
  if (!iam) {
    int *buf1 = (int *)malloc(sizeof(int) * (n + 1));
    int *buf2 = (int *)malloc(sizeof(int) * (A.nnz + A.nnz));

    csr_matrix AAT = addTransposeEliminateDiag(A, buf1, buf2);
    nnz = AAT.nnz;
    MPI_Bcast(&nnz, sizeof(int), MPI_BYTE, 0, comm);

    int *loc_ns, *displs, *loc_nnzs;
    loc_ns = (int *)malloc(sizeof(int) * (np));
    displs = (int *)malloc(sizeof(int) * (np));
    loc_nnzs = (int *)malloc(sizeof(int) * (np));

    vtxdist = (int *)malloc(sizeof(int) * (np + 1));
    for (int i = 0; i <= np; i++)
      vtxdist[i] = (int)(1ll * n * i / np);
    loc_n = vtxdist[iam + 1] - vtxdist[iam];

    rowptr_loc = (int *)malloc(sizeof(int) * (loc_n + 1));

    for (int i = 0; i < np; i++)
      loc_ns[i] = vtxdist[i + 1] - vtxdist[i] + 1;
    MPI_Scatterv(AAT.rowptr, loc_ns, vtxdist, MPI_INT, rowptr_loc, loc_n + 1,
                 MPI_INT, 0, comm);

    for (int i = loc_n; i >= 0; i--)
      rowptr_loc[i] -= rowptr_loc[0];
    loc_nnz = rowptr_loc[loc_n];
    colidx_loc = (int *)malloc(sizeof(int) * loc_nnz);

    for (int i = 0; i < np; i++)
      displs[i] = *(AAT.rowptr + vtxdist[i]);
    for (int i = 0; i < np; i++)
      loc_nnzs[i] = *(AAT.rowptr + vtxdist[i + 1]) - *(AAT.rowptr + vtxdist[i]);
    MPI_Scatterv(AAT.colidx, loc_nnzs, displs, MPI_INT, colidx_loc, loc_nnz,
                 MPI_INT, 0, comm);

    ParMETIS_V3_NodeND(vtxdist, rowptr_loc, colidx_loc, &numflag, options,
                       order, sizes, &comm);

    for (int i = 1; i < np; i++) {
      MPI_Recv(order + vtxdist[i], loc_ns[i], MPI_INT, i, 0, comm,
               MPI_STATUS_IGNORE);
    }

    free(loc_nnzs);
    free(displs);
    free(loc_ns);
    free(vtxdist);
    free(colidx_loc);
    free(rowptr_loc);
    free(buf1);
    free(buf2);
  } else {
    MPI_Bcast(&nnz, sizeof(int), MPI_BYTE, 0, comm);
    vtxdist = (int *)malloc(sizeof(int) * (np + 1));
    for (int i = 0; i <= np; i++)
      vtxdist[i] = (int)(1ll * n * i / np);
    loc_n = vtxdist[iam + 1] - vtxdist[iam];

    rowptr_loc = (int *)malloc(sizeof(int) * (loc_n + 1));

    MPI_Scatterv(nullptr, 0, 0, MPI_INT, rowptr_loc, loc_n + 1, MPI_INT, 0,
                 comm);

    for (int i = loc_n; i >= 0; i--)
      rowptr_loc[i] -= rowptr_loc[0];
    loc_nnz = rowptr_loc[loc_n];
    colidx_loc = (int *)malloc(sizeof(int) * loc_nnz);

    MPI_Scatterv(nullptr, 0, 0, MPI_INT, colidx_loc, loc_nnz, MPI_INT, 0, comm);

    ParMETIS_V3_NodeND(vtxdist, rowptr_loc, colidx_loc, &numflag, options,
                       order, sizes, &comm);

    MPI_Send(order, loc_n, MPI_INT, 0, 0, comm);

    free(vtxdist);
    free(colidx_loc);
    free(rowptr_loc);
  }
}