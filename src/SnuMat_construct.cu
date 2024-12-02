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
void SnuMat::construct_all(csr_matrix A_csr, int *sizes, int *order, double *b) {
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

///////////////////Distribute//////////////////////////void calculate_block(int *sizes, int np)
void SnuMat::self_submatrix(int i, int j) {
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
void SnuMat::send_submatrix(int proc, int i, int j) {

  int loc_nnz = grid(i, j).nnz;

  MPI_Send(&loc_nnz, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
  MPI_Send(grid(i, j).row, loc_nnz, MPI_INT, proc, 0, MPI_COMM_WORLD);
  MPI_Send(grid(i, j).col, loc_nnz, MPI_INT, proc, 0, MPI_COMM_WORLD);
}

void SnuMat::receive_submatrix(int i, int j) {
  int nnz;
  MPI_Recv(&nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  int b = rs_cur;
  rs_cur += nnz;
  max_nnz = max(max_nnz, nnz);

  A[rs_idx] = {block_size[i], block_size[j], nnz,
               loc_r + b,     loc_c + b,     loc_val + b};
  Amap[{i, j}] = rs_idx;
  rs_idx++;

  MPI_Recv(loc_r + b, nnz, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(loc_c + b, nnz, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (int idx = 0; idx < nnz; idx++)
    loc_r[b + idx] -= block_start[i];
  for (int idx = 0; idx < nnz; idx++)
    loc_c[b + idx] -= block_start[j];
}

void SnuMat::set_LU(int i, int j) {
  auto &e = LU[{i, j}];
  auto &M = A[Amap[{i, j}]];
  if (!M.nnz)
    return;

  double *bias = MM_buf + block_start[j] - leaf_size +
                 (block_start[i] - leaf_size) * dense_row;
  for (int idx = 0; idx < M.nnz; idx++) {
    //*(e.data + (M.col[idx] + M.row[idx] * e.m)) += M.data[idx];
    //*(bias + (M.col[idx] + M.row[idx] * e.m)) -= M.data[idx];
    *(MM_buf + (M.col[idx] + block_start[j] - leaf_size) * dense_row +
      (M.row[idx] + block_start[i] - leaf_size)) += M.data[idx];
  }

  // if (0 && e.able) {
  //   gpuErrchk(cudaMemcpy(gpu_row_buf, M.row, M.nnz * sizeof(int),
  //                        cudaMemcpyHostToDevice));
  //   gpuErrchk(cudaMemcpy(gpu_col_buf, M.col, M.nnz * sizeof(int),
  //                        cudaMemcpyHostToDevice));
  //   gpuErrchk(cudaMemcpy(gpu_data_buf, M.data, M.nnz * sizeof(double),
  //                        cudaMemcpyHostToDevice));

  //   int threadsPerBlock = 256;
  //   int blocksPerGrid = (M.nnz + threadsPerBlock - 1) / threadsPerBlock;
  //   copySparseToDense<<<blocksPerGrid, threadsPerBlock>>>(
  //       gpu_row_buf, gpu_col_buf, gpu_data_buf, e.data_gpu, e.n, M.nnz);
  // }
}
void SnuMat::set_all_LU(int i) {
  if (i >= np)
    return;
  set_LU(i, i);
  for (int j = i / 2; j >= 1; j /= 2) {
    set_LU(i, j);
    set_LU(j, i);
  }
}

void SnuMat::send_data_merge_async(int i, MPI_Request *t) {
  MPI_Isend(grid(i, i).data, merge_size[i], MPI_DOUBLE, who[i], i, MPI_COMM_WORLD, t);
}
void SnuMat::send_data_a() {
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
    if (who[i]) {
      send_data_merge_async(i, request + i);
    }
  }

  free(send_order);

  for (auto &i : (my_block)) {
    memcpy(loc_val + merge_start[i], grid(i, i).data,
           merge_size[i] * sizeof(double));
    set_all_LU(i);
  }

  for (int i = num_block; i >= 1; i--) {
    if (who[i])
      MPI_Wait(request + i, MPI_STATUS_IGNORE);
  }
}
void SnuMat::get_data_a_async(int e) {
  MPI_Irecv(loc_val + merge_start[e], merge_size[e], MPI_DOUBLE, 0, e, MPI_COMM_WORLD,
            request + e);
}
void SnuMat::get_data_a() {
  for (auto &i : (my_block))
    get_data_a_async(i);
  for (auto &i : (my_block)) {
    MPI_Wait(request + i, MPI_STATUS_IGNORE);
    set_all_LU(i);
  }

  // for(int i=0; i<merge_size[leaf]; i++) printf("%lf ", loc_val[merge_start[leaf]+i]); printf("\n");
}
void SnuMat::send_data_b() {
  std::fill(_b, _b + local_b_rows, 0.0);
  for (int i = num_block; i >= 1; i--) {
    if (!who[i]) {
      memcpy(_b + block_start[i], perm_b + old_block_start[i],
             block_size[i] * sizeof(double));
    } else {
      MPI_Send(perm_b + old_block_start[i], block_size[i], MPI_DOUBLE, who[i],
               0, MPI_COMM_WORLD);
    }
  }
}
void SnuMat::get_data_b() {
  std::fill(_b, _b + local_b_rows, 0.0);
  for (auto &i : (my_block)) {
    MPI_Recv((b)[i].data, (b)[i].n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}
void SnuMat::malloc_LU(int i, int j, int lvl) {
  int row = block_size[i];
  int col = block_size[j];

  double *data, *data_gpu = nullptr;
  data = (double *)malloc(row * col * sizeof(double));
  if (lvl <= offlvl && (!(iam & 1))) {
    gpuErrchk(cudaMalloc((void **)&data_gpu, row * col * sizeof(double)));
    LU[{i, j}] = {row, col, data, data_gpu, 1};
  } else {
    LU[{i, j}] = {row, col, data, data_gpu, 0};
  }  
}
void SnuMat::free_LU(int i, int j, int lvl) {
  free(LU[{i, j}].data);
  if (lvl <= offlvl && (!(iam & 1)))
    cudaFree(LU[{i, j}].data_gpu);
}
void SnuMat::clear_LU(int i, int j) {
  auto &e = LU[{i, j}];
  int row = e.n;
  int col = e.m;

  std::fill(e.data, e.data + row * col, 0.0);
  if (e.able)
    cudaMemset(e.data_gpu, 0, row * col * sizeof(double));
}

__global__ void copySparseToDense(const int *row, const int *col,
                                  const double *data, double *dmat, int r,
                                  int nnz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nnz) {
    *(dmat + (*(col + idx) * r + *(row + idx))) += *(data + idx);
  }
}

void SnuMat::malloc_all_LU() {
  for (auto &i : (all_parents)) {
    if (i >= np)
      continue;
    int lvl = level[i];
    malloc_LU(i, i, lvl);
    for (int j = i / 2; j >= 1; j /= 2) {
      malloc_LU(i, j, lvl);
      malloc_LU(j, i, lvl);
    }
  }
}
void SnuMat::free_all_LU() {
  for (auto &i : (all_parents)) {
    if (i >= np)
      continue;
    int lvl = level[i];
    free_LU(i, i, lvl);
    for (int j = i / 2; j >= 1; j /= 2) {
      free_LU(i, j, lvl);
      free_LU(j, i, lvl);
    }
  }
}
void SnuMat::malloc_all_b() {
  int sum = 0;
  for (auto &i : (all_parents))
    sum += block_size[i];
  local_b_rows = sum;

  _b = (double *)malloc(sizeof(double) * sum);
  if (offlvl >= 0 && (!(iam & 1))) {
    gpuErrchk(cudaMalloc((void **)&_b_gpu, sum * sizeof(double)));
  }

  sum = 0;
  for (auto &i : (all_parents)) {
    int sz = block_size[i];
    b[i] = {sz, 1, _b + sum, _b_gpu + sum, false};
    sum += sz;
  }
}
void SnuMat::free_all_b() {
  free(_b);
  cudaFree(_b_gpu);
}

void SnuMat::clear_all_LU() {
  for (auto &i : (all_parents)) {
    if (i >= np)
      continue;
    clear_LU(i, i);
    for (int j = i / 2; j >= 1; j /= 2) {
      clear_LU(i, j);
      clear_LU(j, i);
    }
  }

  
  std::fill(MM_buf, MM_buf + dense_row * dense_row, 0);
  if ((!(iam & 1)))
    cudaMemset(MM_buf_gpu, 0, dense_row * dense_row * sizeof(double));
}

void SnuMat::distribute_matrix() {
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

    MPI_Scatter(mat_pp, 1, MPI_INT, &mat_cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nnz_pp, 1, MPI_INT, &loc_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    free(nnz_pp);
    free(mat_pp);
  } else {
    MPI_Scatter(nullptr, 0, 0, &mat_cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nullptr, 0, 0, &loc_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
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
  
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  malloc_all_LU();
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  malloc_all_b();
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  core_preprocess();
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  clear_all_LU();
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  if (!iam) {
    send_data_b();
    send_data_a();
  } else {
    get_data_b();
    get_data_a();
  }

  if (offlvl >= 0 && (!(iam & 1))) {
    max_nnz = max(max_nnz, n);
    // gpuErrchk(cudaMalloc((void **)&gpu_row_buf, max_nnz * sizeof(int)));
    // gpuErrchk(cudaMalloc((void **)&gpu_col_buf, max_nnz * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&gpu_data_buf, max_nnz * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&gpu_data_buf, max_nnz * sizeof(double)));
  }
}

int SnuMat::_who(int block) {
  while (block >= np + np)
    block /= 2;
  while (block < np)
    block = block + block + 1;
  return np + np - block - 1;
}


void SnuMat::make_who() {
  max_level=0;
  for (int i = np; i > 1; i /= 2)
    max_level++;

  vector<int> level(np+np, 0);

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
    if (vst[i]) {
      all_parents.push_back(i);
    }
}
