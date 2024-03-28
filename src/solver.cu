#include "kernel.h"
#include "kernel_gpu.h"
#include "mpi.h"
#include "snusolver.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>

#include <chrono>

static std::chrono::time_point<std::chrono::system_clock> sss, eee;

#define START1() sss = std::chrono::system_clock::now();
#define END1() eee = std::chrono::system_clock::now();
#define GET1()                                                                  \
  (std::chrono::duration_cast<std::chrono::duration<float>>(                   \
       (eee = std::chrono::system_clock::now()) - sss)                         \
       .count())

std::chrono::time_point<std::chrono::system_clock> s, e;

#define START() s = std::chrono::system_clock::now();
#define END() e = std::chrono::system_clock::now();
#define GET()                                                                  \
  (std::chrono::duration_cast<std::chrono::duration<float>>(                   \
       (e = std::chrono::system_clock::now()) - s)                             \
       .count())

using namespace std;

static int np, n, nnz;
static MPI_Comm comm = MPI_COMM_WORLD;
static MPI_Request *request;

// static int iam;
// #define gpuErrchk(ans) \
//   { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line,
//                       bool abort = true) {
//   if (code != cudaSuccess) {
//     fprintf(stderr, "%d: GPUassert: %s %s %d\n", iam,
//     cudaGetErrorString(code),
//             file, line);
//     if (abort)
//       exit(code);
//   }
// }

static int *coo_r, *coo_c;
static double *coo_val;

static int num_block, max_level;
static int *perm_map;
static int *block_start, *old_block_start, *block_size, *send_order;

static const int LEVEL = 9;
static const int NP = 1 << LEVEL;
static int level[NP + NP];

static coo_matrix L[NP + NP][LEVEL + 1];
static coo_matrix U[NP + NP][LEVEL + 1]; // grid[i][j] = A[i][i>>j];

static int who[NP + NP];

//////////////////////////////////////
static const double eps = 1e-15;
static int offlvl = -1;

static vector<int> my_block, my_block_level[LEVEL + 1];
static int *merge_start, *merge_size;
static vector<int> all_parents;

static int mat_cnt, loc_nnz, max_nnz, leaf, leaf_size, local_b_rows;
static int rs_idx, rs_cur;

static int *loc_r, *loc_c;
static double *loc_val;

static coo_matrix *A;
static map<pair<int, int>, int> Amap;

static int dense_row;

static map<pair<int, int>, dense_matrix> LU;

static double *_b;
static map<int, dense_matrix> b;
static double *_b_gpu;

static int core_n, core_m, core_nnz, LU_nnz;
static int *core_rowptr, *core_colidx, *core_map;
static double *core_data;
static int *LU_rowptr, *LU_colidx, *LU_diag, *LU_bias, *LU_map;
static int *LU_rowptr_gpu, *LU_colidx_gpu, *LU_diag_gpu, *LU_bias_gpu,
    *LU_map_gpu;

static int *LU_rowidx_trans, *LU_colptr_trans, *LU_trans_map, *LU_diag_trans;
static int *LU_rowidx_trans_gpu, *LU_colptr_trans_gpu, *LU_trans_map_gpu,
    *LU_diag_trans_gpu;

static int *L_rowidx_trans, *L_colptr_trans, *L_trans_bt;
static int *L_rowidx_trans_gpu, *L_colptr_trans_gpu, *L_trans_bt_gpu;

static double *LU_data, *LU_data_gpu, *MM_buf, *MM_buf2, *MM_buf_gpu;

static cublasHandle_t handle;
static cusolverDnHandle_t cusolverHandle;

static double *LU_buf;
static int *LU_buf_int;
static int *gpu_row_buf, *gpu_col_buf;
static double *gpu_data_buf, *LU_buf_gpu;
static int *order, *sizes;
//////////////////////////////////////

#include "solver_part2.h"

static int _who(int block) {
  while (block >= np + np)
    block /= 2;
  while (block < np)
    block = block + block + 1;
  return np + np - block - 1;
}
static void make_who() {
  max_level = 0;
  for (int i = np; i > 1; i /= 2)
    max_level++;
  for (int i = 1; i <= num_block; i++)
    who[i] = _who(i);
  for (int i = num_block; i >= 1; i--)
    if (who[i] == iam)
      my_block.push_back(i);
  for (int e = 1; e <= num_block; e++) {
    for (int i = e; i > 1; i /= 2)
      level[e]++;
  }
  for (auto &e : my_block) {
    int l = 0;
    my_block_level[level[e]].push_back(e);
  }

  vector<int> vst(num_block + 1, 0);
  for (int i = 1; i <= num_block; i++) {
    if (who[i] == iam) {
      for (int j = i; j >= 1; j /= 2)
        vst[j] = 1;
    }
  }
  for (int i = num_block; i >= 1; i--)
    if (vst[i])
      all_parents.push_back(i);
}

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

  int cur = 0;
  for (auto &e : block_order) {
    block_start[e] = cur;
    block_size[e] = sizes[np + np - 1 - e];
    cur += block_size[e];
  }
}

static csr_matrix permutate(csr_matrix A, int *order) { //& construct
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

void construct_all(csr_matrix A_csr, int *sizes, int *_order, double *b) {
  order = _order;

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

void self_submatrix(int i, int j) {
  int nnz = grid(i, j).nnz;

  int b = rs_cur;
  rs_cur += nnz;

  A[rs_idx] = {block_size[i], block_size[j], nnz,
               loc_r + b,     loc_c + b,     loc_val + b};
  Amap[{i, j}] = rs_idx;
  rs_idx++;

  memcpy(loc_r + b, grid(i, j).row, nnz * sizeof(int));
  memcpy(loc_c + b, grid(i, j).col, nnz * sizeof(int));

  for (int idx = 0; idx < nnz; idx++)
    loc_r[b + idx] -= block_start[i];
  for (int idx = 0; idx < nnz; idx++)
    loc_c[b + idx] -= block_start[j];
}
void send_submatrix(int proc, int i, int j) {

  int loc_nnz = grid(i, j).nnz;

  MPI_Send(&loc_nnz, 1, MPI_INT, proc, 0, comm);
  MPI_Send(grid(i, j).row, loc_nnz, MPI_INT, proc, 0, comm);
  MPI_Send(grid(i, j).col, loc_nnz, MPI_INT, proc, 0, comm);
}

static void receive_submatrix(int i, int j) {
  int nnz;
  MPI_Recv(&nnz, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);

  int b = rs_cur;
  rs_cur += nnz;
  max_nnz = max(max_nnz, nnz);

  A[rs_idx] = {block_size[i], block_size[j], nnz,
               loc_r + b,     loc_c + b,     loc_val + b};
  Amap[{i, j}] = rs_idx;
  rs_idx++;

  MPI_Recv(loc_r + b, nnz, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
  MPI_Recv(loc_c + b, nnz, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);

  for (int idx = 0; idx < nnz; idx++)
    loc_r[b + idx] -= block_start[i];
  for (int idx = 0; idx < nnz; idx++)
    loc_c[b + idx] -= block_start[j];
}

static void send_data_merge_async(int i, MPI_Request *t) {
  MPI_Isend(grid(i, i).data, merge_size[i], MPI_DOUBLE, who[i], i, comm, t);
}
static void send_data_a() {
  int cnt = 0;

  send_order = (int *)malloc(sizeof(int) * (np * 2));
  vector<vector<int>> v;
  v.resize(np);
  for (int i = num_block; i >= 1; i--)
    v[who[i]].push_back(i);
  for (int k = 0; k < (int)(v[0].size()); k++) {
    for (int p = 0; p < np; p++) {
      if (int(v[p].size()) <= k)
        continue;
      send_order[cnt++] = v[p][k];
    }
  }

  for (int idx = 0; idx < num_block; idx++) {
    int i = send_order[idx];

    // for (int i = num_block; i >= 1; i--) {
    if (who[i])
      send_data_merge_async(i, request + i);
  }

  free(send_order);

  for (auto &i : my_block) {
    memcpy(loc_val + merge_start[i], grid(i, i).data,
           merge_size[i] * sizeof(double));
    set_all_LU(i);
  }

  for (int i = num_block; i >= 1; i--) {
    if (who[i])
      MPI_Wait(request + i, MPI_STATUS_IGNORE);
  }
}
void get_data_a_async(int e) {
  MPI_Irecv(loc_val + merge_start[e], merge_size[e], MPI_DOUBLE, 0, e, comm,
            request + e);
}
static void get_data_a() {
  for (auto &i : my_block)
    get_data_a_async(i);
  for (auto &i : my_block) {
    MPI_Wait(request + i, MPI_STATUS_IGNORE);
    set_all_LU(i);
  }
}

static void send_data_b() {
  std::fill(_b, _b + local_b_rows, 0.0);
  for (int i = num_block; i >= 1; i--) {
    if (!who[i]) {
      memcpy(_b + block_start[i], perm_b + old_block_start[i],
             block_size[i] * sizeof(double));
    } else {
      MPI_Send(perm_b + old_block_start[i], block_size[i], MPI_DOUBLE, who[i],
               0, comm);
    }
  }
}

void get_data_b() {
  std::fill(_b, _b + local_b_rows, 0.0);
  for (auto &i : my_block) {
    MPI_Recv(b[i].data, b[i].n, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
  }
}

void distribute_all() {
  make_who();
  request = (MPI_Request *)malloc(sizeof(MPI_Request) * (num_block + 1));
  if (!iam) {
    int *mat_pp = (int *)malloc(sizeof(int) * np);
    int *nnz_pp = (int *)malloc(sizeof(int) * np);
    for (int i = 0; i < np; i++)
      mat_pp[i] = 0, nnz_pp[i] = 0;

    for (int i = num_block; i >= 1; i--) {
      int p = who[i];
      mat_pp[p]++;
      nnz_pp[p] += grid(i, i).nnz;
      for (int ii = i / 2; ii; ii /= 2) {
        mat_pp[p] += 2;
        nnz_pp[p] += grid(i, ii).nnz;
        nnz_pp[p] += grid(ii, i).nnz;
      }
    }

    MPI_Scatter(mat_pp, 1, MPI_INT, &mat_cnt, 1, MPI_INT, 0, comm);
    MPI_Scatter(nnz_pp, 1, MPI_INT, &loc_nnz, 1, MPI_INT, 0, comm);

    free(nnz_pp);
    free(mat_pp);
  } else {
    MPI_Scatter(nullptr, 0, 0, &mat_cnt, 1, MPI_INT, 0, comm);
    MPI_Scatter(nullptr, 0, 0, &loc_nnz, 1, MPI_INT, 0, comm);
  }

  loc_r = (int *)malloc(sizeof(int) * loc_nnz);
  loc_c = (int *)malloc(sizeof(int) * loc_nnz);
  loc_val = (double *)malloc(sizeof(double) * loc_nnz);
  A = (coo_matrix *)malloc(sizeof(coo_matrix) * mat_cnt);

  max_nnz = 0;
  rs_idx = 0;
  rs_cur = 0;

  if (!iam) {
    for (int i = num_block; i >= 1; i--) {
      int p = who[i];
      if (who[i]) {
        send_submatrix(p, i, i);
        for (int ii = i / 2; ii; ii /= 2) {
          send_submatrix(p, i, ii);
          send_submatrix(p, ii, i);
        }
      } else {
        merge_start[i] = rs_cur;
        self_submatrix(i, i);
        for (int ii = i / 2; ii; ii /= 2) {
          self_submatrix(i, ii);
          self_submatrix(ii, i);
        }
        merge_size[i] = rs_cur - merge_start[i];
      }
    }
  } else {
    for (auto &i : my_block) {
      merge_start[i] = rs_cur;
      receive_submatrix(i, i);
      for (int ii = i / 2; ii; ii /= 2) {
        receive_submatrix(i, ii);
        receive_submatrix(ii, i);
      }
      merge_size[i] = rs_cur - merge_start[i];
    }
  }

  leaf = np + np - 1 - iam;
  leaf_size = block_size[leaf];

  local_b_rows = 0;
  for (auto &i : all_parents)
    local_b_rows += block_size[i];
  dense_row = local_b_rows - leaf_size;

  malloc_all_LU();
  malloc_all_b();
  core_preprocess();

  clear_all_LU();

  if (!iam) {
    send_data_b();
    send_data_a();
  } else {
    get_data_b();
    get_data_a();
  }

  if (offlvl >= 0 && (!(iam & 1))) {
    max_nnz = max(max_nnz, n);
    gpuErrchk(cudaMalloc((void **)&gpu_row_buf, max_nnz * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&gpu_col_buf, max_nnz * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&gpu_data_buf, max_nnz * sizeof(double)));
  }
}
static void gather_data_b() {
  for (int i = num_block; i >= 1; i--) {
    if (!who[i]) {
      memcpy(perm_b + old_block_start[i], b[i].data,
             block_size[i] * sizeof(double));
    } else {
      MPI_Recv(perm_b + old_block_start[i], block_size[i], MPI_DOUBLE, who[i],
               0, comm, MPI_STATUS_IGNORE);
    }
  }
}

void return_data_b() {
  for (auto &i : my_block) {
    MPI_Send(b[i].data, b[i].n, MPI_DOUBLE, 0, 0, comm);
  }
}

void factsolve(double *b_ret) {

  core_run();

  START()
  for (int l = max_level - 1; l > max(offlvl, -1); l--) {
    for (auto &i : my_block_level[l]) {
      snusolver_LU(LU[{i, i}]);
      snusolver_trsm_Lxb(LU[{i, i}], b[i]);

      for (int j = i / 2; j; j /= 2) {
        snusolver_trsm_xUb(LU[{i, i}], LU[{j, i}]);
        snusolver_trsm_Lxb(LU[{i, i}], LU[{i, j}]);
      }

      for (int j1 = i / 2; j1; j1 /= 2) {
        for (int j2 = i / 2; j2; j2 /= 2) {
          snusolver_gemm(LU[{j1, i}], LU[{i, j2}], LU[{j1, j2}]);
        }
      }

      for (int j = i / 2; j; j /= 2) {
        snusolver_gemm(LU[{j, i}], b[i], b[j]);
      }
    }
    for (auto &i : my_block_level[l]) {
      reduction(i);
    }
  }

  if ((!iam))
    cout << "\t" << iam << " Dense LU " << GET() << endl;

  START()
  if (offlvl >= 0 && (!(iam & 1))) {
    b_togpu();
    for (auto &i : all_parents) {
      LU[{i, i}].toGPU();
      for (int j = i / 2; j >= 1; j /= 2) {
        LU[{i, j}].toGPU();
        LU[{j, i}].toGPU();
      }
    }
  }
  cudaDeviceSynchronize();

  if ((!iam))
    cout << "\t" << iam << " Memcpy " << GET() << endl;
  START()
  for (int l = min(max_level - 1, offlvl); l >= 0; l--) {
    for (auto &i : my_block_level[l]) {
      snusolver_LU_gpu(LU[{i, i}], cusolverHandle);
      snusolver_trsm_Lxb_gpu(LU[{i, i}], b[i], handle);

      for (int j = i / 2; j; j /= 2) {
        snusolver_trsm_xUb_gpu(LU[{i, i}], LU[{j, i}], handle);
        snusolver_trsm_Lxb_gpu(LU[{i, i}], LU[{i, j}], handle);
      }

      for (int j1 = i / 2; j1; j1 /= 2) {
        for (int j2 = i / 2; j2; j2 /= 2) {
          snusolver_gemm_gpu(LU[{j1, i}], LU[{i, j2}], LU[{j1, j2}], handle);
        }
      }

      for (int j = i / 2; j; j /= 2) {
        snusolver_gemm_gpu(LU[{j, i}], b[i], b[j], handle);
      }
    }
    for (auto &i : my_block_level[l]) {
      reduction_gpu(i);
    }
  }
  if ((!iam))
    cout << "\t" << iam << " Dense LU gpu " << GET() << endl;

  START()
  for (int l = 0; l <= min(max_level - 1, offlvl); l++) {
    for (int idx = my_block_level[l].size() - 1; idx >= 0; idx--) {
      int i = my_block_level[l][idx];
      scatter_b_gpu(i);

      for (int j = i / 2; j >= 1; j /= 2) {
        snusolver_gemm_gpu(LU[{i, j}], b[j], b[i], handle);
      }
      snusolver_trsm_Uxb_gpu(LU[{i, i}], b[i], handle);
    }
  }
  if ((!iam))
    cout << "\t" << iam << " Dense solve gpu " << GET() << endl;

  if (offlvl >= 0 && (!(iam & 1)))
    b_tocpu();
  START()
  for (int l = max(offlvl + 1, 0); l < max_level; l++) {
    for (int idx = my_block_level[l].size() - 1; idx >= 0; idx--) {
      int i = my_block_level[l][idx];
      scatter_b(i);

      for (int j = i / 2; j >= 1; j /= 2) {
        snusolver_gemm(LU[{i, j}], b[j], b[i]);
      }
      snusolver_trsm_Uxb(LU[{i, i}], b[i]);
    }
  }
  if ((!iam))
    cout << "\t" << iam << " Dense solve " << GET() << endl;

  START() {
    scatter_b(leaf);
    // // TODODO
    // START()
    // int blockSize = 256;
    // int gridSize = (leaf_size + blockSize - 1) / blockSize;

    // kernel4_1<<<gridSize, blockSize>>>(_b_gpu, LU_rowptr_gpu, LU_colidx_gpu,
    //                                    LU_data_gpu, LU_bias_gpu, LU_diag_gpu,
    //                                    leaf_size);
    // cudaDeviceSynchronize();
    // if ( (! iam) && 0) cout << "\t" << iam << " step4-1! " << GET() << endl;
    // START()
    // kernel4_2<<<1, 32>>>(_b_gpu, LU_rowptr_gpu, LU_colidx_gpu, LU_data_gpu,
    //                      LU_bias_gpu, LU_diag_gpu, LU_colptr_trans_gpu,
    //                      LU_rowidx_trans_gpu, LU_diag_trans_gpu,
    //                      LU_trans_map_gpu, leaf_size);
    // cudaDeviceSynchronize();

    for (int r = 0; r < leaf_size; r++) {
      for (int ptr = LU_rowptr[r + 1] - 1; ptr >= LU_bias[r]; ptr--) {
        int c = LU_colidx[ptr];
        _b[r] -= LU_data[ptr] * _b[c];
      }
    }
    if ((!iam))
      cout << "\t" << iam << " Sparse solve 1 " << GET() << endl;

    START()
    for (int r = leaf_size - 1; r >= 0; r--) {
      for (int ptr = LU_diag[r] + 1; ptr < LU_bias[r]; ptr++) {
        int c = LU_colidx[ptr];
        _b[r] -= LU_data[ptr] * _b[c];
      }
      _b[r] /= LU_data[LU_diag[r]];
    }
    if ((!iam))
      cout << "\t" << iam << " Sparse solve 2 " << GET() << endl;
  }

  if (!iam) {
    gather_data_b();
    for (int i = 0; i < n; i++)
      b_ret[i] = perm_b[order[i]];
  } else {
    return_data_b();
  }
}

void createHandle() {
  cublasCreate(&handle);
  cusolverDnCreate(&cusolverHandle);
}

void solve(csr_matrix A_csr, double *b, double *x) {
  MPI_Comm_rank(comm, &iam);
  MPI_Comm_size(comm, &np);

  if (!iam) {
    n = A_csr.n, nnz = A_csr.nnz;
  }

  MPI_Bcast(&n, sizeof(int), MPI_BYTE, 0, comm);
  MPI_Bcast(&nnz, sizeof(int), MPI_BYTE, 0, comm);

  A_csr.n = n, A_csr.nnz = nnz;

  sizes = (int *)malloc(sizeof(int) * (np * 2 - 1));
  order = (int *)malloc(sizeof(int) * n);

          
          MPI_Barrier(comm); START1();
  call_parmetis(A_csr, sizes, order);
          //MPI_Barrier(comm); if(!iam) cout << "parmetis: " << GET1() << endl; START1();
  construct_all(A_csr, sizes, order, b);
          //MPI_Barrier(comm); if(!iam) cout << "construct: " << GET1() << endl; START1();
  distribute_all();
          //MPI_Barrier(comm); if(!iam) cout << "distribute: " << GET1() << endl; START1();
  factsolve(x);
          MPI_Barrier(comm); if(!iam) cout << "fact: " << GET1() << endl; START1();
          MPI_Barrier(comm); if(!iam) cout << "total: " << GET1() << endl; START1();

  free(sizes);
  free(order);
}