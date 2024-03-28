#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "kernel.h"
std::chrono::time_point<std::chrono::system_clock> s, e;

#define START() s = std::chrono::system_clock::now();
#define END() e = std::chrono::system_clock::now();
#define GET()                                                                  \
  (std::chrono::duration_cast<std::chrono::duration<float>>(                   \
       (e = std::chrono::system_clock::now()) - s)                             \
       .count())

static int n, m, nnz, LU_nnz;
static int *rowptr, *colidx;
static double *data;
static int *LU_rowptr, *LU_colidx, *LU_diag;
static double *LU_data;

double eps = 1e-16;

using namespace std;

static void dump(const char *filename, void *data, int len) {
  ofstream ofile;
  ofile.open(filename, std::ofstream::out | std::ofstream::binary);
  ofile.write(reinterpret_cast<char *>(data), len);
  ofile.close();
}
void symbfact() {
  // based on S.C.EISENSTATANDJ.W.H.LIU

  int MAX_NNZ = nnz * 100;

  int *new_rowptr = (int *)malloc(sizeof(int) * (n + 1));
  int *new_colidx = (int *)malloc(sizeof(int) * (MAX_NNZ));

  vector<int> cur_fsnz(n, n);
  vector<int> vst(n, 0);

  new_rowptr[0] = 0;
  int cur = 0;

  int *Uptr = (int *)malloc(sizeof(int) * n);
  int *Lj = (int *)malloc(sizeof(int) * n);
  int *Uj = (int *)malloc(sizeof(int) * n);
  int cntL, cntU;
  for (int j = 0; j < n; j++) {
    cntL = 0, cntU = 0;
    for (int i = rowptr[j]; i < rowptr[j + 1]; i++) {
      int h = colidx[i];
      if (h < j)
        Lj[cntL++] = h, vst[h] = 1;
      else
        Uj[cntU++] = h, vst[h] = 1;
    }
    for (int idx = 0; idx < cntL; idx++) {
      int i = Lj[idx];
      for (int uidx = Uptr[i]; uidx < new_rowptr[i + 1]; uidx++) {
        int h = new_colidx[uidx];
        // if(h>cur_fsnz[i]) continue;
        if (h < j) {
          if (!vst[h])
            Lj[cntL++] = h, vst[h] = 1;
        } else {
          if (!vst[h])
            Uj[cntU++] = h, vst[h] = 1;
        }

        if (h == j)
          cur_fsnz[i] = j;
      }
    }

    sort(Lj, Lj + cntL);
    sort(Uj, Uj + cntU);

    if (cntL + cntU + cur > MAX_NNZ) {
      MAX_NNZ += nnz;
      realloc(new_colidx, MAX_NNZ + nnz);
    }

    for (int i = 0; i < cntL; i++)
      new_colidx[cur++] = Lj[i];
    Uptr[j] = cur;

    for (int i = 0; i < cntU; i++)
      new_colidx[cur++] = Uj[i];
    new_rowptr[j + 1] = cur;

    for (int i = 0; i < cntL; i++)
      vst[Lj[i]] = false;
    for (int i = 0; i < cntU; i++)
      vst[Uj[i]] = false;
  }

  LU_colidx = new_colidx;
  LU_rowptr = new_rowptr;
  LU_diag = Uptr;
  LU_nnz = cur;

  LU_data = (double *)malloc(sizeof(double) * LU_nnz);
  // printf("symbfact Done!"); fflush(stdout);
  free(Lj);
  free(Uj);
}
static void setLU() {
  for (int i = 0; i < LU_nnz; i++)
    LU_data[i] = 0.0;
  for (int r = 0; r < n; r++) {
    int ptr = LU_rowptr[r];
    for (int ci = rowptr[r]; ci < rowptr[r + 1]; ci++) {
      int c = colidx[ci];

      while (LU_colidx[ptr] < c)
        ptr++;
      LU_data[ptr] = data[ci];
    }
  }
}
void numfact() {
  setLU();
  int cnt = 0;
  for (int r = 0; r < n; r++) {
    for (int cur_ptr = LU_rowptr[r]; cur_ptr < LU_rowptr[r + 1]; cur_ptr++) {
      int cur_c = LU_colidx[cur_ptr];
      if (cur_c >= r)
        break;
      double d = LU_data[cur_ptr] / LU_data[LU_diag[cur_c]];

      int cur_itr = cur_ptr;
      LU_data[cur_ptr] /= LU_data[LU_diag[cur_c]];
      for (int prv_ptr = LU_diag[cur_c] + 1; prv_ptr < LU_rowptr[cur_c + 1];
           prv_ptr++) {
        int prv_c = LU_colidx[prv_ptr];
        while (LU_colidx[cur_itr] < prv_c) {
          cur_itr++;
        }
        LU_data[cur_itr] -= LU_data[prv_ptr] * d;
      }
    }
    if (LU_data[LU_diag[r]] < eps && LU_data[LU_diag[r]] >= 0)
      LU_data[LU_diag[r]] = eps, cnt++;
    if (LU_data[LU_diag[r]] > -eps && LU_data[LU_diag[r]] < 0)
      LU_data[LU_diag[r]] = -eps, cnt++;
  }
}

void snusolver_LU_sparse(coo_matrix &M, csr_matrix &LU) {
  n = M.n, m = M.m, nnz = M.nnz;
  rowptr = (int *)malloc(sizeof(int) * (n + 1));
  colidx = (int *)malloc(sizeof(int) * nnz);
  data = (double *)malloc(sizeof(double) * nnz);

  for (int i = 0; i < n + 1; i++)
    rowptr[i] = 0;
  for (int i = 0; i < nnz; i++)
    rowptr[M.row[i]]++;
  for (int i = 0; i < n; i++)
    rowptr[i + 1] += rowptr[i];

  for (int i = nnz - 1; i >= 0; i--) {
    int r = M.row[i];
    colidx[--rowptr[r]] = M.col[i];
    data[rowptr[r]] = M.data[i];
  }

  symbfact();
  numfact();
  // START();
  // symbfact();
  // END();
  // auto t1=GET();
  // START();
  // numfact();
  // END();
  // auto t2=GET();

  // std::cout << t1 << " " << t2 << std::endl;

  LU.n = n, LU.m = m, LU.nnz = LU_nnz;
  LU.rowptr = LU_rowptr, LU.colidx = LU_colidx, LU.data = LU_data;
}

void snusolver_trsm_Lxb_sparse(csr_matrix &L, dense_matrix &b, int i, int j) {
  // Row major
  int n = b.n, m = b.m;
  if (!n || !m)
    return;
  for (int c = 0; c < m; c++) {
    for (int r = 0; r < n; r++) {
      for (int ptr = L.rowptr[r]; ptr < L.rowptr[r + 1]; ptr++) {
        int cur_c = L.colidx[ptr];

        if (cur_c >= r)
          break;

        b.data[r * m + c] -= L.data[ptr] * b.data[cur_c * m + c];
      }
    }
  }
}
void snusolver_trsm_xUb_sparse(csr_matrix &U, dense_matrix &b, int i, int j) {
  int n = b.n, m = b.m;
  if (!n || !m)
    return;
  for (int r = 0; r < n; r++) {
    for (int c = 0; c < m; c++) {
      for (int ptr = U.rowptr[c]; ptr < U.rowptr[c + 1]; ptr++) {
        int cur_c = U.colidx[ptr];

        if (cur_c < c)
          continue; // Diag부터 시작하도록
        if (cur_c == c) {
          b.data[r * m + c] /= U.data[ptr];
        } else {
          b.data[r * m + cur_c] -= U.data[ptr] * b.data[r * m + c];
        }
      }
    }
  }
}
void snusolver_trsm_Uxb_sparse(csr_matrix &U, dense_matrix &b, int i, int j) {
  int n = b.n, m = b.m;
  if (!n || !m)
    return;

  for (int c = 0; c < m; c++) {
    for (int r = n - 1; r >= 0; r--) {
      double d;
      for (int ptr = U.rowptr[r]; ptr < U.rowptr[r + 1]; ptr++) {
        int cur_c = U.colidx[ptr];

        if (cur_c < r)
          continue; // Diag부터 시작하도록
        if (cur_c == r) {
          d = U.data[ptr];
        } else
          b.data[r * m + c] -= U.data[ptr] * b.data[cur_c * m + c];
      }
      b.data[r * m + c] /= d;
    }
  }
}