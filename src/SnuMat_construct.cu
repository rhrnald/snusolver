#include <algorithm>
#include "mpi.h"

#include "SnuMat.h"

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

///////////////construct////////////////////

void SnuMat::dfs_block_order(vector<int> &v, int cur, int parts) {
  if (parts > 1) {
    dfs_block_order(v, cur * 2 + 1, parts / 2);
    dfs_block_order(v, cur * 2, parts / 2);
  }
  v.push_back(cur);
}
void SnuMat::calculate_block(int *sizes, int np) {
  vector<int> block_order;
  dfs_block_order(block_order, 1, np);

  int cur = 0;
  for (auto &e : block_order) {
    block_start[e] = cur;
    block_size[e] = sizes[np + np - 1 - e];
    cur += block_size[e];
  }
}

csr_matrix SnuMat::permutate(csr_matrix A, int *order) { //& construct
  int *indices = A.colidx;
  int *indptr = A.rowptr;
  int rows = A.n, cols = A.m, nnz = A.nnz;

  int sum = 0;

  int *new_indptr = (int *)malloc(sizeof(int) * (A.n + 1));
  int *new_indices = (int *)malloc(sizeof(int) * (A.nnz));

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
void SnuMat::clearGrid(int i, int j) {
  coo_matrix &M = grid(i, j);
  M.n = block_size[i];
  M.m = block_size[j];
  M.nnz = 0;
}
int SnuMat::setGrid(int i, int j, int bias) {
  coo_matrix &M = grid(i, j);
  M.data = coo_val + bias;
  bias += M.nnz;
  M.row = coo_r + bias;
  M.col = coo_c + bias;
  return bias;
}
int SnuMat::addElement(int i, int j, int r, int c) {
  coo_matrix &M = grid(i, j);

  M.row--;
  M.col--;
  *M.row = r;
  *M.col = c;

  return M.row - coo_r;
}
void SnuMat::construct(csr_matrix A) {
  int *indices = A.colidx;
  int *indptr = A.rowptr;
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

  for (int r = 0; r < rows; r++) {
    for (int idx = indptr[r]; idx < indptr[r + 1]; idx++) {
      int c = indices[idx];
      grid(block[r], block[c]).nnz++;
    }
  }

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
void SnuMat::construct_structure(csr_matrix A_csr, int *sizes, int *order, double *b) {
  num_block = np + np - 1;
  merge_start = (int *)malloc(sizeof(int) * (num_block + 1));
  merge_size = (int *)malloc(sizeof(int) * (num_block + 1));

  block_start = (int *)malloc(sizeof(int) * (np * 2));
  block_size = (int *)malloc(sizeof(int) * (np * 2));
  calculate_block(sizes, np);
  if (!iam) {
    perm_map = (int *)malloc(sizeof(int) * nnz);
    coo_r = (int *)malloc(sizeof(int) * nnz);
    coo_c = (int *)malloc(sizeof(int) * nnz);
    coo_val = (double *)malloc(sizeof(double) * nnz);

    csr_matrix PA = permutate(A_csr, order);
    construct(PA);

    for (int i = 0; i < nnz; i++)
      coo_val[perm_map[i]] = A_csr.data[i];

    old_block_start = (int *)malloc(sizeof(int) * (np * 2));
    memcpy(old_block_start, block_start, sizeof(int) * (np * 2));

    perm_b = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
      perm_b[order[i]] = b[i];
    }
  }
}
