#include "snusolver.h"
#include <vector>
#include <algorithm>
#include "mpi.h"
static MPI_Comm comm = MPI_COMM_WORLD;

using namespace std;

static int np, iam, n, nnz;

static int *coo_r, *coo_c;
static double *coo_val;

static int num_block;
static int *perm_map;
static int *block_start, *block_size, *merge_size, *send_order;

static const int LEVEL = 9;
static const int NP = 1 << LEVEL;

static coo_matrix L[NP+NP][LEVEL + 1];
static coo_matrix U[NP+NP][LEVEL + 1]; // grid[i][j] = A[i][i>>j];

static int who[NP+NP];

static double *perm_b;

coo_matrix &grid(int i, int j) {
  // assert max/2^k = min
  int x = max(i, j), y = min(i, j);
  int k = 0;
  for (int jj = x; jj > y; jj /= 2)
    k++;
  if (i >= j)
    return U[x][k];
  else
    return L[x][k];
}

static void dfs_block_order(vector<int> &v, int cur, int parts) {
  if (parts > 1) {
    dfs_block_order(v, cur * 2 + 1, parts / 2);
    dfs_block_order(v, cur * 2, parts / 2);
  }
  v.push_back(cur);
}
static void calculate_block(int *sizes, int np) {
  vector<int> block_order;
  dfs_block_order(block_order, 1, np);
  num_block = np+np - 1;

  int cur = 0;
  for (auto &e : block_order) {
    block_start[e] = cur;
    block_size[e] = sizes[np+np - 1 - e];
    cur += block_size[e];
  }
}

static csr_matrix permutate(csr_matrix A, int *order) { //& construct
  int *indices = A.colidx;
  int *indptr = A.rowptr;
  int rows = A.n, cols = A.m, nnz = A.nnz;

  int sum = 0;

  int *new_indptr = (int*)malloc(sizeof(int)*(A.n+1));
  int *new_indices = (int*)malloc(sizeof(int)*(A.nnz));

  new_indptr[0] = 0;
  for (int i = 0; i < cols; i++)
    new_indptr[order[i] + 1] = indptr[i + 1] - indptr[i];

  int max_row = 0;
  for (int i = 0; i < cols; i++)
    max_row = max(max_row, new_indptr[i + 1]);
  for (int i = 0; i < cols; i++)
    new_indptr[i + 1] += new_indptr[i];

  for (int i = 0; i < nnz; i++)
    sum += indices[i];

  vector<pair<int, int>> v(max_row);
  for (int i = 0; i < cols; i++) {
    // ith row *(rowptr+i)~*(rowptr+i+1)
    // goes to new_rowptr[order[i]]~new_rowptr[order[i]+1]
    int bs = indptr[i + 1] - indptr[i];
    v.resize(bs);
    for (int j = indptr[i]; j < indptr[i + 1]; j++) {
      v[j - indptr[i]] = {order[indices[j]], j};
    }
    sort(v.begin(), v.begin() + bs);
    for (int j = new_indptr[order[i]], jj = 0; j < new_indptr[order[i] + 1];
         j++, jj++) {
      new_indices[j] = order[indices[v[jj].second]];
      perm_map[v[jj].second] = j;
    }
  }
  return {rows, cols, nnz, new_indptr, new_indices, nullptr};
}
static void clearGrid(int i, int j) {
  coo_matrix &M = grid(i, j);
  M.n = block_size[i];
  M.m = block_size[j];
  M.nnz = 0;
}
static int setGrid(int i, int j, int bias) {
  coo_matrix &M = grid(i, j);
  M.data = coo_val + bias;
  bias += M.nnz;
  M.row = coo_r + bias;
  M.col = coo_c + bias;
  return bias;
}
static int addElement(int i, int j, int r, int c) {
  coo_matrix &M = grid(i, j);

  M.row--;
  M.col--;
  *M.row = r;
  *M.col = c;

  return M.row - coo_r;
}
static void construct(csr_matrix A) {
  int *indices = A.rowptr;
  int *indptr = A.colidx;
  int rows = A.n, cols = A.m, nnz = A.nnz;
  int *block = (int *)malloc(sizeof(int) * cols);
  int *construct_map = (int *)malloc(sizeof(int) * nnz);

  for (int b = 1; b <= num_block; b++) {
    for (int i = block_start[b]; i < block_start[b] + block_size[b]; i++) {
      block[i] = b;
    }
  }

  for (int i = num_block; i >= 1; i--) {
    clearGrid(i, i);
    for (int ii = i / 2; ii; ii /= 2) {
      clearGrid(i, ii);
      clearGrid(ii, i);
    }
  }

  // for (int c = 0; c < cols; c++) {
  //   for (int idx = indptr[c]; idx < indptr[c + 1]; idx++) {
  //     int r = indices[idx];
  //     grid(block[r], block[c]).nnz++;
  //   }
  // }
  
  for (int r = 0; r < rows; r++) {
    for (int idx = indptr[r]; idx < indptr[r + 1]; idx++) {
      int c = indices[idx];
      printf("%d %d\n", r, c);
      grid(block[r], block[c]).nnz++;
    }
  }

   return;
  int bias = 0;
  for (int i = num_block; i >= 1; i--) {
    merge_size[i] = bias;
    bias = setGrid(i, i, bias);
    for (int ii = i / 2; ii; ii /= 2) {
      bias = setGrid(i, ii, bias);
      bias = setGrid(ii, i, bias);
    }
  }
  merge_size[0] = bias;
  for (int i = num_block; i >= 1; i--)
    merge_size[i] = merge_size[i - 1] - merge_size[i];

  // for (int c = cols - 1; c >= 0; c--) {
  //   for (int idx = indptr[c + 1] - 1; idx >= indptr[c]; idx--) {
  //     int r = indices[idx];
  //     construct_map[idx] = addElement(block[r], block[c], r, c);
  //   }
  // }
  for (int r = rows - 1; r >= 0; r--) {
    for (int idx = indptr[r + 1] - 1; idx >= indptr[r]; idx--) {
      int c = indices[idx];
      construct_map[idx] = addElement(block[r], block[c], r, c);
    }
  }

  for (int i = 0; i < nnz; i++)
    perm_map[i] = construct_map[perm_map[i]];

  free(construct_map);
  free(block);
}

int construct_all(csr_matrix A_csr, int *sizes, int *order){
   
  MPI_Comm_rank(comm, &iam);
  MPI_Comm_size(comm, &np);
  n=A_csr.n, nnz=A_csr.nnz;

  block_start = (int *)malloc(sizeof(int) * (np * 2-1));
  block_size = (int *)malloc(sizeof(int) * (np * 2-1));
  perm_map = (int *)malloc(sizeof(int) * nnz);
  coo_r = (int *)malloc(sizeof(int) * nnz);
  coo_c = (int *)malloc(sizeof(int) * nnz);
  coo_val = (double *)malloc(sizeof(double) * nnz);

   calculate_block(sizes, np);
   csr_matrix PA = permutate(A_csr, order);
   // construct(PA);
}
