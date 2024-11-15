static long long sparse_flop_getrf=0;
static long long sparse_flop_trsm=0;
static long long sparse_flop_gemm=0;

static void malloc_LU(int i, int j, int lvl) {
  int row = block_size[i];
  int col = block_size[j];

  double *data, *data_gpu = nullptr;
  data = (double *)malloc(row * col * sizeof(double));
  if (lvl <= offlvl && (!(iam & 1)))
    gpuErrchk(cudaMalloc((void **)&data_gpu, row * col * sizeof(double)));

  LU[{i, j}] = {row, col, data, data_gpu, lvl <= offlvl};
}
static void free_LU(int i, int j, int lvl) {
  free(LU[{i, j}].data);
  if (lvl <= offlvl && (!(iam & 1)))
    cudaFree(LU[{i, j}].data_gpu);
}

static void clear_LU(int i, int j) {
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

static void set_LU(int i, int j) {
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

  if (0 && e.able) {
    gpuErrchk(cudaMemcpy(gpu_row_buf, M.row, M.nnz * sizeof(int),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_col_buf, M.col, M.nnz * sizeof(int),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_data_buf, M.data, M.nnz * sizeof(double),
                         cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (M.nnz + threadsPerBlock - 1) / threadsPerBlock;
    copySparseToDense<<<blocksPerGrid, threadsPerBlock>>>(
        gpu_row_buf, gpu_col_buf, gpu_data_buf, e.data_gpu, e.n, M.nnz);
  }
}
void dense_matrix::toCPU() {
  cudaMemcpy(data, data_gpu, n * m * sizeof(double), cudaMemcpyDeviceToHost);
}
void dense_matrix::toGPU() {
  // if (able)
  cudaMemcpy(data_gpu, data, n * m * sizeof(double), cudaMemcpyHostToDevice);
}
static void malloc_all_LU() {
  for (auto &i : all_parents) {
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
static void free_all_LU() {
  for (auto &i : all_parents) {
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
static void malloc_all_b() {
  int sum = 0;
  for (auto &i : all_parents)
    sum += block_size[i];
  local_b_rows = sum;

  _b = (double *)malloc(sizeof(double) * sum);
  if (offlvl >= 0 && (!(iam & 1))) {
    gpuErrchk(cudaMalloc((void **)&_b_gpu, sum * sizeof(double)));
  }

  sum = 0;
  for (auto &i : all_parents) {
    int sz = block_size[i];
    b[i] = {sz, 1, _b + sum, _b_gpu + sum, false};
    sum += sz;
  }
}
static void free_all_b() {
  free(_b);
  cudaFree(_b_gpu);
}

static void clear_all_LU() {
  for (auto &i : all_parents) {
    if (i >= np)
      continue;
    clear_LU(i, i);
    for (int j = i / 2; j >= 1; j /= 2) {
      clear_LU(i, j);
      clear_LU(j, i);
    }
  }

  std::fill(MM_buf, MM_buf + dense_row * dense_row, 0);
  cudaMemset(MM_buf_gpu, 0, dense_row * dense_row * sizeof(double));
}
static void set_all_LU(int i) {
  // for(auto &i: my_block) {
  if (i >= np)
    return;
  set_LU(i, i);
  for (int j = i / 2; j >= 1; j /= 2) {
    set_LU(i, j);
    set_LU(j, i);
  }
  //
}

void core_togpu() {
  gpuErrchk(cudaMemcpy(LU_data_gpu, LU_data, (LU_nnz) * sizeof(double),
                       cudaMemcpyHostToDevice));
}
void b_togpu() {
  gpuErrchk(cudaMemcpy(_b_gpu, _b, local_b_rows * sizeof(double),
                       cudaMemcpyHostToDevice));
}
void b_tocpu() {
  gpuErrchk(cudaMemcpy(_b, _b_gpu, local_b_rows * sizeof(double),
                       cudaMemcpyDeviceToHost));
}

void reduction(int block_num) { // Todo change to MPI_REDUCE & non blocking
  if (block_num & 1) {
    int src = who[block_num - 1];
    if (src == iam)
      return;

    for (int j1 = block_num / 2; j1; j1 /= 2) {
      for (int j2 = block_num / 2; j2; j2 /= 2) {
        auto &M = LU[{j1, j2}];
        int cnt = M.n * M.m;
        double *tmp = (double *)malloc(sizeof(double) * cnt);
        MPI_Recv(tmp, cnt, MPI_DOUBLE, src, j1 * num_block + j2, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        // for(int idx=0; idx<cnt; idx++) tmp[idx]+=tmp2[idx];
        for (int i = 0; i < cnt; i++)
          M.data[i] += tmp[i];

        free(tmp);
      }

      int cnt = b[j1].n;
      double *tmp = (double *)malloc(sizeof(double) * cnt);
      MPI_Recv(tmp, cnt, MPI_DOUBLE, src, j1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      for (int idx = 0; idx < cnt; idx++)
        b[j1].data[idx] += tmp[idx];

      free(tmp);
    }
  } else {
    int dst = who[block_num + 1];
    if (dst == iam)
      return;

    for (int j1 = block_num / 2; j1; j1 /= 2) {
      for (int j2 = block_num / 2; j2; j2 /= 2) {
        auto &M = LU[{j1, j2}];
        int cnt = M.n * M.m;
        MPI_Send(M.data, cnt, MPI_DOUBLE, dst, j1 * num_block + j2,
                 MPI_COMM_WORLD);
      }

      int cnt = b[j1].n;
      MPI_Send(b[j1].data, cnt, MPI_DOUBLE, dst, j1, MPI_COMM_WORLD);
    }
  }
}
void scatter_b(int block_num) { // Todo change to MPI_REDUCE & non blocking
  if (block_num & 1) {
    int dst = who[block_num - 1];
    if (dst == iam)
      return;
    for (int j = block_num / 2; j >= 1; j /= 2) {
      MPI_Send(b[j].data, b[j].n, MPI_DOUBLE, dst, j, MPI_COMM_WORLD);
    }
  } else {
    int src = who[block_num + 1];
    if (src == iam)
      return;
    for (int j = block_num / 2; j >= 1; j /= 2) {
      MPI_Recv(b[j].data, b[j].n, MPI_DOUBLE, src, j, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  }
}
__global__ void vectorAdd(double *A, const double *B, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) {
    A[i] = A[i] + B[i];
  }
}
void core_reduction() { // Todo change to MPI_REDUCE & non blocking
  if (leaf & 1) {
    int src = who[leaf - 1];
    if (src == iam)
      return;
    MPI_Recv(MM_buf2, dense_row * dense_row, MPI_DOUBLE, src, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    for (int i = 0; i < dense_row * dense_row; i++) {
      MM_buf[i] += MM_buf2[i];
    }
    // for (int j1 = leaf / 2; j1; j1 /= 2) {
    //   int cnt = b[j1].n;
    //   double *tmp = (double *) malloc(sizeof(double) * cnt);
    //   MPI_Recv(tmp, cnt, MPI_DOUBLE, src, j1, MPI_COMM_WORLD,
    //            MPI_STATUS_IGNORE);
    //   for (int i = 0; i < cnt; i++) b[j1].data[i] += tmp[i];
    //   free(tmp);
    // }
    int cnt = dense_row;
    double *tmp = (double *)malloc(sizeof(double) * cnt);
    MPI_Recv(tmp, cnt, MPI_DOUBLE, src, leaf / 2, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    for (int i = 0; i < cnt; i++)
      b[leaf / 2].data[i] += tmp[i];
    free(tmp);

  } else {
    int dst = who[leaf + 1];
    if (dst == iam)
      return;
    MPI_Send(MM_buf, dense_row * dense_row, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD);

    int cnt = dense_row;
    MPI_Send(b[leaf / 2].data, cnt, MPI_DOUBLE, dst, leaf / 2, MPI_COMM_WORLD);
  }
}
void reduction_gpu(int block_num) { // Todo change to MPI_REDUCE & non blocking
  if (block_num & 1) {
    int src = who[block_num - 1];
    if (src == iam)
      return;

    for (int j1 = block_num / 2; j1; j1 /= 2) {
      for (int j2 = block_num / 2; j2; j2 /= 2) {
        auto &M = LU[{j1, j2}];
        int cnt = M.n * M.m;
        int blockSize = 256;
        int gridSize = (cnt + blockSize - 1) / blockSize;

        MPI_Recv(MM_buf_gpu, cnt * sizeof(double), MPI_BYTE, src,
                 j1 * num_block + j2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        vectorAdd<<<gridSize, blockSize>>>(M.data_gpu, MM_buf_gpu, cnt);
        cudaDeviceSynchronize();
      }
    }
    int cnt = core_n - block_start[block_num / 2];
    int blockSize = 256;
    int gridSize = (cnt + blockSize - 1) / blockSize;

    MPI_Recv(gpu_data_buf, cnt, MPI_DOUBLE, src, block_num / 2, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    vectorAdd<<<gridSize, blockSize>>>(b[block_num / 2].data_gpu, gpu_data_buf,
                                       cnt);
    cudaDeviceSynchronize();
  } else {
    int dst = who[block_num + 1];
    if (dst == iam)
      return;

    cudaDeviceSynchronize();
    for (int j1 = block_num / 2; j1; j1 /= 2) {
      for (int j2 = block_num / 2; j2; j2 /= 2) {
        auto &M = LU[{j1, j2}];
        int cnt = M.n * M.m;
        MPI_Send(M.data_gpu, cnt * sizeof(double), MPI_BYTE, dst,
                 j1 * num_block + j2, MPI_COMM_WORLD);
      }
    }
    int cnt = core_n - block_start[block_num / 2];
    MPI_Send(b[block_num / 2].data_gpu, cnt, MPI_DOUBLE, dst, block_num / 2,
             MPI_COMM_WORLD);
  }
}
void scatter_b_gpu(int block_num) { // Todo change to MPI_REDUCE & non blocking
  if (block_num & 1) {
    int dst = who[block_num - 1];
    if (dst == iam)
      return;
    for (int j = block_num / 2; j >= 1; j /= 2) {
      int cnt = b[j].n;
      double *tmp = (double *)malloc(sizeof(double) * cnt);
      gpuErrchk(cudaMemcpy(tmp, b[j].data_gpu, cnt * sizeof(double),
                           cudaMemcpyDeviceToHost));
      MPI_Send(tmp, b[j].n, MPI_DOUBLE, dst, j, MPI_COMM_WORLD);
      free(tmp);
      // MPI_Send(b[j].data_gpu, cnt, MPI_DOUBLE, dst, j, MPI_COMM_WORLD);
    }
    // int cnt = core_n-block_start[block_num/2];
    // MPI_Send(b[block_num/2].data_gpu, cnt, MPI_DOUBLE, dst, block_num/2,
    // MPI_COMM_WORLD);
  } else {
    int src = who[block_num + 1];
    if (src == iam)
      return;
    for (int j = block_num / 2; j >= 1; j /= 2) {
      int cnt = b[j].n;

      double *tmp = (double *)malloc(sizeof(double) * cnt);
      MPI_Recv(tmp, b[j].n, MPI_DOUBLE, src, j, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      gpuErrchk(cudaMemcpy(b[j].data_gpu, tmp, cnt * sizeof(double),
                           cudaMemcpyHostToDevice));
      free(tmp);

      // MPI_Recv(b[j].data_gpu, cnt, MPI_DOUBLE, src, j, MPI_COMM_WORLD,
      //       MPI_STATUS_IGNORE);
    }

    // int cnt = core_n-block_start[block_num/2];
    // int blockSize = 256;
    // int gridSize = (cnt + blockSize - 1) / blockSize;
    // MPI_Recv(gpu_data_buf, cnt, MPI_DOUBLE, src, block_num/2, MPI_COMM_WORLD,
    //           MPI_STATUS_IGNORE);
    // vectorAdd<<<gridSize, blockSize>>>(b[block_num/2].data_gpu, gpu_data_buf,
    // cnt); cudaDeviceSynchronize();
  }
}

__global__ void updateLU(double *LU_data, int *LU_rowptr, int *LU_colidx,
                         int *LU_diag, int leaf_size, int core_n) {
  int r = blockIdx.x * blockDim.x + threadIdx.x + leaf_size;
  if (r < core_n) {
    for (int cur_ptr = LU_rowptr[r]; cur_ptr < LU_rowptr[r + 1]; cur_ptr++) {
      int cur_c = LU_colidx[cur_ptr];
      if (cur_c >= leaf_size)
        break;
      double d = LU_data[cur_ptr] / LU_data[LU_diag[cur_c]];

      int cur_itr = cur_ptr;
      LU_data[cur_ptr] /= LU_data[LU_diag[cur_c]];
      for (int prv_ptr = LU_diag[cur_c] + 1; prv_ptr < LU_rowptr[cur_c + 1];
           prv_ptr++) {
        int prv_c = LU_colidx[prv_ptr];
        if (prv_c >= leaf_size)
          break;
        while (LU_colidx[cur_itr] < prv_c) {
          cur_itr++;
        }
        LU_data[cur_itr] -= LU_data[prv_ptr] * d;
      }
    }
  }
}
__global__ void sparseLU_v2(double *LU_data, int *LU_rowptr, int *LU_colidx,
                            int *LU_diag, int *L_colptr_trans,
                            int *L_rowidx_trans, int *L_trans_bt, int leaf_size,
                            int core_n) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int N = gridDim.x * blockDim.x;
  for (int r = 0; r < leaf_size; r++) {
    __syncthreads();

    double tmp = LU_data[LU_diag[r]];
    if (tmp < eps && tmp > -eps) {
      if (tmp < 0)
        tmp = -eps;
      else
        tmp = eps;
    }
    if (n == 0) {
      LU_data[LU_diag[r]] = tmp;
    }
    __syncthreads();

    for (int cur_ptr = L_colptr_trans[r] + n; cur_ptr < L_colptr_trans[r + 1];
         cur_ptr += N) {
      int nxt_r = L_rowidx_trans[cur_ptr];

      int nxt_itr = L_trans_bt[cur_ptr];
      double d = LU_data[nxt_itr] / tmp;
      LU_data[nxt_itr] /= tmp;

      for (int cur_itr = LU_diag[r] + 1; cur_itr < LU_rowptr[r + 1];
           cur_itr++) {
        int cur_c = LU_colidx[cur_itr];
        while (LU_colidx[nxt_itr] < cur_c) {
          nxt_itr++;
        }
        LU_data[nxt_itr] -= LU_data[cur_itr] * d;
      }
    }
  }
}

__global__ void sparseLU_1(double *LU_data, int *LU_rowptr, int *LU_colidx,
                           int *LU_diag, int *L_colptr_trans,
                           int *L_rowidx_trans, int *L_trans_bt, int leaf_size,
                           int core_n, int r) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int N = gridDim.x * blockDim.x;
  // for (int r = 0; r < leaf_size; r++) {
  //   __syncthreads();

  double tmp = LU_data[LU_diag[r]];
  if (tmp < eps && tmp > -eps) {
    if (tmp < 0)
      tmp = -eps;
    else
      tmp = eps;
  }
  if (n == 0) {
    LU_data[LU_diag[r]] = tmp;
  }
  // __syncthreads();

  // for (int cur_ptr = L_colptr_trans[r] + n; cur_ptr < L_colptr_trans[r + 1];
  //     cur_ptr += N) {
  //   int nxt_r = L_rowidx_trans[cur_ptr];

  //   int nxt_itr = L_trans_bt[cur_ptr];
  //   double d = LU_data[nxt_itr] / tmp;
  //   LU_data[nxt_itr] /= tmp;

  //   for (int cur_itr = LU_diag[r] + 1; cur_itr < LU_rowptr[r + 1]; cur_itr++)
  //   {
  //     int cur_c = LU_colidx[cur_itr];
  //     while (LU_colidx[nxt_itr] < cur_c) { nxt_itr++; }
  //     LU_data[nxt_itr] -= LU_data[cur_itr] * d;
  //   }
  // }
  //}
}
__global__ void sparseLU_2(double *LU_data, int *LU_rowptr, int *LU_colidx,
                           int *LU_diag, int *L_colptr_trans,
                           int *L_rowidx_trans, int *L_trans_bt,
                           double *LU_buf_gpu, int leaf_size, int core_n,
                           int r) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int N = gridDim.x * blockDim.x;
  double tmp = LU_data[LU_diag[r]];
  for (int cur_ptr = L_colptr_trans[r] + n; cur_ptr < L_colptr_trans[r + 1];
       cur_ptr += N) {
    int nxt_r = L_rowidx_trans[cur_ptr];
    int nxt_itr = L_trans_bt[cur_ptr];
    LU_data[nxt_itr] /= tmp;
    LU_buf_gpu[nxt_r] = LU_data[nxt_itr];
  }
}
__global__ void sparseLU_4(double *LU_data, int *LU_rowptr, int *LU_colidx,
                           int *LU_diag, int *L_colptr_trans,
                           int *L_rowidx_trans, int *L_trans_bt,
                           double *LU_buf_gpu, int leaf_size, int core_n,
                           int r) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int N = gridDim.x * blockDim.x;
  double tmp = LU_data[LU_diag[r]];
  for (int cur_ptr = L_colptr_trans[r] + n; cur_ptr < L_colptr_trans[r + 1];
       cur_ptr += N) {
    int nxt_r = L_rowidx_trans[cur_ptr];
    LU_buf_gpu[nxt_r] = 0;
  }
}
__global__ void sparseLU_3(double *LU_data, int *LU_rowptr, int *LU_colidx,
                           int *LU_diag, int *LU_colptr_trans,
                           int *LU_rowidx_trans, int *LU_trans_map, int *LU_map,
                           double *LU_buf_gpu, int leaf_size, int core_n,
                           int r) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int N = gridDim.x * blockDim.x;
  // for (int r = 0; r < leaf_size; r++) {
  //   __syncthreads();

  // double tmp = LU_data[LU_diag[r]];
  // if (tmp < eps && tmp > -eps) {
  //   if (tmp < 0)
  //     tmp = -eps;
  //   else
  //     tmp = eps;
  // }
  // if (n == 0) { LU_data[LU_diag[r]] = tmp; }
  // __syncthreads();

  double tmp = LU_data[LU_diag[r]];

  // for (int cur_ptr = L_colptr_trans[r] + n; cur_ptr < L_colptr_trans[r + 1];
  //     cur_ptr += N) {
  //   int nxt_r = L_rowidx_trans[cur_ptr];

  //   int nxt_itr = L_trans_bt[cur_ptr];

  //   for (int cur_itr = LU_diag[r] + 1; cur_itr < LU_rowptr[r + 1]; cur_itr++)
  //   {
  //     int cur_c = LU_colidx[cur_itr];
  //     while (LU_colidx[nxt_itr] < cur_c) { nxt_itr++; }
  //     LU_data[nxt_itr] -= LU_data[cur_itr] * LU_buf_gpu[nxt_r];
  //   }
  // }

  for (int cur_itr = LU_diag[r] + 1 + blockIdx.x; cur_itr < LU_rowptr[r + 1];
       cur_itr += gridDim.x) {
    int cur_c = LU_colidx[cur_itr];
    for (int cur_ptr = LU_map[cur_itr] + 1 + threadIdx.x;
         cur_ptr < LU_colptr_trans[cur_c + 1]; cur_ptr += blockDim.x) {
      int nxt_r = LU_rowidx_trans[cur_ptr];
      int nxt_ptr = LU_trans_map[cur_ptr];
      LU_data[nxt_ptr] -= LU_data[cur_itr] * LU_buf_gpu[nxt_r];
    }
  }
  //}
}

__global__ void sparseTRSM_kernel(double *_b, int *LU_rowptr, int *LU_colidx,
                                  int *LU_diag, double *LU_data,
                                  int *L_colptr_trans, int *L_rowidx_trans,
                                  int *L_trans_bt_gpu, int leaf_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = gridDim.x * blockDim.x;
  // if (r < leaf_size) {
  int sum = 0;
  for (int r = 0; r < leaf_size; r++) {
    __syncthreads();
    double d = _b[r];
    for (int ptr = L_colptr_trans[r] + idx; ptr < L_colptr_trans[r + 1];
         ptr += N) {
      int r2 = L_rowidx_trans[ptr];
      int bt = L_trans_bt_gpu[ptr];
      _b[r2] -= d * LU_data[bt];
      // sum+=r2;
    }
    /*double tmp = 0;
    for (int ptr = LU_rowptr_gpu[r]; ptr < LU_diag_gpu[r]; ptr++) {
      int c = LU_colidx_gpu[ptr];
      tmp += LU_data_gpu[ptr] * _b[c];
    }
    _b[r] -= tmp;*/
  }
}
__global__ void copyArray(int *LU_itr_gpu, int *LU_bias_gpu, int core_n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < core_n) {
    LU_itr_gpu[k] = LU_bias_gpu[k];
  }
}
__global__ void sparseSPMM_kernel(double *data, double *LU_data, int *LU_colidx,
                                  int L, int R, int m, int M, int ptr1,
                                  int bias, int n) {
  int ptr2 = blockIdx.x * blockDim.x + threadIdx.x + L;
  if (ptr2 >= R)
    return;
  int c = LU_colidx[ptr2];
  if (c < m || c >= M)
    return;

  data[bias + c * n] -= LU_data[ptr1] * LU_data[ptr2];
}
__global__ void sparseSPMM_kernel2(double *MM_buf, int *LU_rowptr,
                                   int *LU_colidx, double *LU_data,
                                   int *LU_bias, int leaf_size, int core_n,
                                   int dense_row) {
  int r = blockIdx.x * blockDim.x + threadIdx.x + leaf_size;
  int bias = r - (dense_row + 1) * leaf_size;
  if (r >= core_n)
    return;
  // for(int r=leaf_size; r<core_n; r++) {
  for (int ptr1 = LU_rowptr[r]; ptr1 < LU_rowptr[r + 1]; ptr1++) {
    int r2 = LU_colidx[ptr1];
    double val = LU_data[ptr1];

    for (int ptr2 = LU_bias[r2]; ptr2 < LU_rowptr[r2 + 1]; ptr2++) {
      int c = LU_colidx[ptr2];
      double val2 = LU_data[ptr2];
      MM_buf[c * dense_row + bias] += val * val2;
    }
  }
  //}
}
__global__ void updateData(double *data, double *MM_buf_gpu, int r1, int r2,
                           int c1, int c2, int dense_row, int leaf_size) {
  int c = blockIdx.x * blockDim.x + threadIdx.x + c1;
  while (c < c2) {
    int r = blockIdx.y * blockDim.y + threadIdx.y + r1;
    while (r < r2) {
      data[(r - r1) + (c - c1) * (r2 - r1)] =
          MM_buf_gpu[(r - leaf_size) + (c - leaf_size) * dense_row];
      r += blockDim.y * gridDim.y; // Move to the next iteration in y-direction
    }
    c += blockDim.x * gridDim.x; // Move to the next iteration in x-direction
  }
}
__global__ void backwardSubstitution(double *_b, int *LU_rowptr, int *LU_colidx,
                                     double *LU_data, int leaf_size,
                                     int core_n) {
  // int r = blockIdx.x * blockDim.x + threadIdx.x+leaf_size;
  // if (r >= leaf_size && r < core_n) {
  for (int r = leaf_size; r < core_n; r++) {
    double tmp = 0;
    for (int ptr = LU_rowptr[r]; ptr < LU_rowptr[r + 1]; ptr++) {
      int c = LU_colidx[ptr];
      tmp += LU_data[ptr] * _b[c];
    }
    _b[r] -= tmp;
  }
}

void core_symbfact() {
  int MAX_NNZ = core_nnz * 100;

  int *new_rowptr = (int *)malloc(sizeof(int) * (core_n + 1));
  int *new_colidx = (int *)malloc(sizeof(int) * (MAX_NNZ));

  // vector<int> cur_fsnz(core_n, core_n);
  vector<int> vst(core_n, 0);

  new_rowptr[0] = 0;
  int cur = 0;

  int *Uptr = (int *)malloc(sizeof(int) * core_n);
  LU_bias = (int *)malloc(sizeof(int) * core_n);
  // LU_itr = (int *) malloc(sizeof(int) * core_n);
  int *Lj = (int *)malloc(sizeof(int) * core_n);
  int *Uj = (int *)malloc(sizeof(int) * core_n);
  int cntL, cntU;
  for (int j = 0; j < leaf_size; j++) {
    cntL = 0, cntU = 0;
    for (int i = core_rowptr[j]; i < core_rowptr[j + 1]; i++) {
      int h = core_colidx[i];
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

        // if(h==j) cur_fsnz[i]=j;
      }
    }

    sort(Lj, Lj + cntL);
    sort(Uj, Uj + cntU);

    if (cntL + cntU + cur > MAX_NNZ) {
      MAX_NNZ <<= 1;
      new_colidx = (int *)realloc(new_colidx, MAX_NNZ * sizeof(int));
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
  for (int j = leaf_size; j < core_n; j++) {
    cntL = 0, cntU = 0;
    for (int i = core_rowptr[j]; i < core_rowptr[j + 1]; i++) {
      int h = core_colidx[i];
      if (h < j)
        Lj[cntL++] = h, vst[h] = 1;
      else
        Uj[cntU++] = h, vst[h] = 1;
    }

    for (int idx = 0; idx < cntL; idx++) {
      int i = Lj[idx];
      for (int uidx = Uptr[i]; uidx < new_rowptr[i + 1]; uidx++) {
        int h = new_colidx[uidx];
        if (h >= leaf_size)
          break;
        // if(h>cur_fsnz[i]) continue;
        if (h < j) {
          if (!vst[h])
            Lj[cntL++] = h, vst[h] = 1;
        } else {
          if (!vst[h])
            Uj[cntU++] = h, vst[h] = 1;
        }
        // if(h==j) cur_fsnz[i]=j;
      }
    }

    sort(Lj, Lj + cntL);
    sort(Uj, Uj + cntU);

    if (cntL + cntU + cur > MAX_NNZ) {
      MAX_NNZ <<= 1;
      new_colidx = (int *)realloc(new_colidx, MAX_NNZ * sizeof(int));
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

  free(Lj);
  free(Uj);

  LU_data = (double *)malloc(sizeof(double) * LU_nnz);
  MM_buf = (double *)malloc(sizeof(double) * dense_row * dense_row);
  MM_buf2 = (double *)malloc(sizeof(double) * dense_row * dense_row);
  LU_buf = (double *)malloc(sizeof(double) * core_n);
  for (int i = 0; i < core_n; i++)
    LU_buf[i] = 0;
  LU_buf_int = (int *)malloc(sizeof(int) * core_n);
  for (int i = 0; i < core_n; i++)
    LU_buf_int[i] = 0;

  if (offlvl >= 0 && (!(iam & 1))) {
    gpuErrchk(cudaMalloc((void **)&LU_rowptr_gpu, (core_n + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&LU_colidx_gpu, (LU_nnz) * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&LU_diag_gpu, (core_n) * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&LU_bias_gpu, (core_n) * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&LU_data_gpu, (LU_nnz) * sizeof(double)));

    gpuErrchk(cudaMalloc((void **)&MM_buf_gpu,
                         dense_row * dense_row * sizeof(double)));

    gpuErrchk(cudaMalloc((void **)&LU_buf_gpu, core_n * sizeof(double)));

    gpuErrchk(cudaMemcpy(LU_rowptr_gpu, LU_rowptr, (core_n + 1) * sizeof(int),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(LU_colidx_gpu, LU_colidx, (LU_nnz) * sizeof(int),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(LU_diag_gpu, LU_diag, (core_n) * sizeof(int),
                         cudaMemcpyHostToDevice));
  }
}

void core_preprocess() {
  int sum = 0;
  for (auto &i : all_parents) {
    block_start[i] = sum;
    sum += block_size[i];
  }
  core_n = sum, core_m = sum;
  //   which_block = (int *)malloc(sizeof(int) * core_n);

  //   for (auto &i : all_parents) {
  //     for (int j = block_start[i]; j < block_start[i] + block_size[i]; j++)
  //       which_block[j] = i;
  //   }

  vector<pair<int, int>> core_blocks;
  core_blocks.push_back({leaf, leaf});
  for (int ii = leaf / 2; ii > 0; ii /= 2) {
    core_blocks.push_back({leaf, ii});
    core_blocks.push_back({ii, leaf});
  }

  core_nnz = 0;
  for (auto &e : core_blocks) {
    core_nnz += A[Amap[e]].nnz;
  }

  core_rowptr = (int *)malloc(sizeof(int) * (core_n + 1));
  core_colidx = (int *)malloc(sizeof(int) * core_nnz);
  core_map = (int *)malloc(sizeof(int) * core_nnz);
  core_data = (double *)malloc(sizeof(double) * core_nnz);

  for (int i = 0; i < core_n + 1; i++)
    core_rowptr[i] = 0;
  for (auto &e : core_blocks) {
    auto &M = A[Amap[e]];
    int i = e.first;

    for (int ptr = 0; ptr < M.nnz; ptr++) {
      core_rowptr[M.row[ptr] + block_start[i]]++;
    }
  }
  for (int i = 0; i < sum; i++)
    core_rowptr[i + 1] += core_rowptr[i];
  for (int idx = core_blocks.size() - 1; idx >= 0; idx--) {
    auto &e = core_blocks[idx];
    auto &M = A[Amap[e]];
    int i = e.first, j = e.second;

    for (int ptr = M.nnz - 1; ptr >= 0; ptr--) {
      int r = M.row[ptr] + block_start[i];
      int c = M.col[ptr] + block_start[j];
      core_colidx[--core_rowptr[r]] = c;
      core_data[core_rowptr[r]] = M.data[ptr];
    }
  }

  START()
  core_symbfact();
  if ((!iam) && 1)
    cout << "\t" << iam << " symbfact : " << GET() << endl;
  for (int i = 0; i < core_n; i++)
    core_rowptr[i] = LU_rowptr[i];
  for (auto &e : core_blocks) {
    auto &M = A[Amap[e]];
    int i = e.first, j = e.second;

    for (int ptr = 0; ptr < M.nnz; ptr++) {
      int r = M.row[ptr] + block_start[i];
      int c = M.col[ptr] + block_start[j];
      while (LU_colidx[core_rowptr[r]] != c)
        core_rowptr[r]++;
      core_map[(M.data - loc_val) + ptr] = core_rowptr[r];
    }
  }

  for (int i = 0; i < leaf_size; i++) {
    int ptr = LU_rowptr[i];
    while (ptr < LU_rowptr[i + 1] && LU_colidx[ptr] < leaf_size) {
      ptr++;
    }
    LU_bias[i] = ptr;
  }

  L_colptr_trans = (int *)malloc((leaf_size + 1) * sizeof(int));

  for (int i = 0; i <= leaf_size; i++)
    L_colptr_trans[i] = 0;
  for (int i = 0; i < leaf_size; i++) {
    for (int ptr = LU_rowptr[i]; ptr < LU_diag[i]; ptr++) {
      L_colptr_trans[LU_colidx[ptr]]++;
    }
  }

  for (int i = 0; i < leaf_size; i++)
    L_colptr_trans[i + 1] += L_colptr_trans[i];

  L_rowidx_trans = (int *)malloc(L_colptr_trans[leaf_size] * sizeof(int));
  L_trans_bt = (int *)malloc(L_colptr_trans[leaf_size] * sizeof(int));
  for (int r = leaf_size - 1; r >= 0; r--) {
    for (int ptr = LU_rowptr[r]; ptr < LU_diag[r]; ptr++) {
      int c = LU_colidx[ptr];
      L_rowidx_trans[--L_colptr_trans[c]] = r;
      L_trans_bt[L_colptr_trans[c]] = ptr;
    }
  }

  LU_colptr_trans = (int *)malloc((core_n + 1) * sizeof(int));
  LU_rowidx_trans = (int *)malloc(LU_nnz * sizeof(int));
  LU_trans_map = (int *)malloc(LU_nnz * sizeof(int));
  LU_map = (int *)malloc(LU_nnz * sizeof(int));
  LU_diag_trans = (int *)malloc(leaf_size * sizeof(int));

  for (int r = 0; r <= core_n; r++)
    LU_colptr_trans[r] = 0;
  for (int r = 0; r < core_n; r++) {
    for (int ptr = LU_rowptr[r]; ptr < LU_rowptr[r + 1]; ptr++) {
      LU_colptr_trans[LU_colidx[ptr]]++;
    }
  }

  for (int i = 0; i < core_n; i++)
    LU_colptr_trans[i + 1] += LU_colptr_trans[i];
  for (int r = core_n - 1; r >= 0; r--) {
    for (int ptr = LU_rowptr[r]; ptr < LU_rowptr[r + 1]; ptr++) {
      int c = LU_colidx[ptr];
      LU_rowidx_trans[--LU_colptr_trans[c]] = r;
      LU_trans_map[LU_colptr_trans[c]] = ptr;
      LU_map[ptr] = LU_colptr_trans[c];
    }
  }

  for (int r = 0; r < leaf_size; r++)
    LU_diag_trans[r] = LU_map[LU_diag[r]];

  if (offlvl >= 0 && (!(iam & 1))) {
    gpuErrchk(cudaMalloc((void **)&L_colptr_trans_gpu,
                         (leaf_size + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&L_rowidx_trans_gpu,
                         (L_colptr_trans[leaf_size]) * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&L_trans_bt_gpu,
                         (L_colptr_trans[leaf_size]) * sizeof(int)));
    gpuErrchk(cudaMemcpy(L_colptr_trans_gpu, L_colptr_trans,
                         (leaf_size + 1) * sizeof(int),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(L_rowidx_trans_gpu, L_rowidx_trans,
                         (L_colptr_trans[leaf_size]) * sizeof(int),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(L_trans_bt_gpu, L_trans_bt,
                         (L_colptr_trans[leaf_size]) * sizeof(int),
                         cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMalloc((void **)&LU_colptr_trans_gpu, (core_n + 1) * sizeof(int)));
    gpuErrchk(
        cudaMalloc((void **)&LU_rowidx_trans_gpu, (LU_nnz) * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&LU_trans_map_gpu, (LU_nnz) * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&LU_map_gpu, (LU_nnz) * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&LU_diag_trans_gpu, leaf_size * sizeof(int)));
    gpuErrchk(cudaMemcpy(LU_colptr_trans_gpu, LU_colptr_trans,
                         (core_n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(LU_rowidx_trans_gpu, LU_rowidx_trans,
                         (LU_nnz) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(LU_trans_map_gpu, LU_trans_map, (LU_nnz) * sizeof(int),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(LU_map_gpu, LU_map, (LU_nnz) * sizeof(int),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(LU_diag_trans_gpu, LU_diag_trans,
                         leaf_size * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(LU_bias_gpu, LU_bias, (core_n) * sizeof(int),
                         cudaMemcpyHostToDevice));
  }
}

void core_numfact_v1() {
  for (int r = 0; r < leaf_size; r++) {
    for (int cur_ptr = LU_rowptr[r]; cur_ptr < LU_diag[r]; cur_ptr++) {
      int cur_c = LU_colidx[cur_ptr];
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
    double tmp = LU_data[LU_diag[r]];
    if (tmp < eps && tmp > -eps) {
      if (tmp < 0)
        LU_data[LU_diag[r]] = -eps;
      else
        LU_data[LU_diag[r]] = eps;
    }
  }

  if ((!iam) && 1)
    cout << "\t" << iam << " step1-1 (First level LU-1) : " << GET() << endl;
  START()
  for (int r = leaf_size; r < core_n; r++) {
    for (int cur_ptr = LU_rowptr[r]; cur_ptr < LU_rowptr[r + 1]; cur_ptr++) {
      int cur_c = LU_colidx[cur_ptr];
      if (cur_c >= leaf_size)
        break;
      double d = LU_data[cur_ptr] / LU_data[LU_diag[cur_c]];

      int cur_itr = cur_ptr;
      LU_data[cur_ptr] /= LU_data[LU_diag[cur_c]];
      for (int prv_ptr = LU_diag[cur_c] + 1; prv_ptr < LU_rowptr[cur_c + 1];
           prv_ptr++) {
        int prv_c = LU_colidx[prv_ptr];
        if (prv_c >= leaf_size)
          break;
        while (LU_colidx[cur_itr] < prv_c) {
          cur_itr++;
        }
        LU_data[cur_itr] -= LU_data[prv_ptr] * d;
      }
    }
  }
  // if (!iam) printf(" -- flop: %lld %lld %lld %lld\n", flop1, idx1, flop2,
  // idx2);
}
void core_numfact_v2() {
  for (int r = 0; r < leaf_size; r++) {
    double tmp = LU_data[LU_diag[r]];
    if (tmp < eps && tmp > -eps) {
      if (tmp < 0)
        tmp = -eps;
      else
        tmp = eps;
    }
    LU_data[LU_diag[r]] = tmp;

    for (int cur_ptr = LU_diag[r] + 1; cur_ptr < LU_rowptr[r + 1]; cur_ptr++) {
      int cur_c = LU_colidx[cur_ptr];
      LU_buf[cur_c] = LU_data[cur_ptr];
    }

    int Lbound = LU_map[LU_diag[r]] + 1;
    int Rbound = LU_colptr_trans[r + 1];

    // #pragma omp parallel for num_threads(8)
    for (int cur_ptr = Lbound; cur_ptr < Rbound; cur_ptr++) {
      int nxt_row = LU_rowidx_trans[cur_ptr];
      int nxt_ptr = LU_trans_map[cur_ptr];

      LU_data[nxt_ptr] /= tmp;
      double dd = LU_data[nxt_ptr];
      int L = LU_rowptr[nxt_row + 1];
      for (int nxt_itr = nxt_ptr + 1; nxt_itr < L; nxt_itr++) {
        int nxt_c = LU_colidx[nxt_itr];
        LU_data[nxt_itr] -= dd * LU_buf[nxt_c];
        sparse_flop_getrf+=2;
      }
    }

    for (int cur_ptr = LU_diag[r] + 1; cur_ptr < LU_rowptr[r + 1]; cur_ptr++) {
      int cur_c = LU_colidx[cur_ptr];
      LU_buf[cur_c] = 0;
    }
  }
}
void core_numfact_v3() {
  for (int r = 0; r < leaf_size; r++) {
    double tmp = LU_data[LU_diag[r]];
    if (tmp < eps && tmp > -eps) {
      if (tmp < 0)
        tmp = -eps;
      else
        tmp = eps;
    }
    LU_data[LU_diag[r]] = tmp;
    // #pragma omp parallel for num_threads(8)
    for (int cur_ptr = LU_map[LU_diag[r]] + 1; cur_ptr < LU_colptr_trans[r + 1];
         cur_ptr++) {
      int nxt_r = LU_rowidx_trans[cur_ptr];
      int nxt_itr = LU_trans_map[cur_ptr];
      LU_data[nxt_itr] /= tmp;
      LU_buf[nxt_r] = LU_data[nxt_itr];
    }
    // #pragma omp parallel for num_threads(8)
    for (int cur_itr = LU_diag[r] + 1; cur_itr < LU_rowptr[r + 1]; cur_itr++) {
      int cur_c = LU_colidx[cur_itr];
      for (int cur_ptr = LU_map[cur_itr] + 1;
           cur_ptr < LU_colptr_trans[cur_c + 1]; cur_ptr++) {
        int nxt_r = LU_rowidx_trans[cur_ptr];
        int nxt_ptr = LU_trans_map[cur_ptr];
        LU_data[nxt_ptr] -= LU_data[cur_itr] * LU_buf[nxt_r];
      }
    }
    for (int cur_ptr = LU_map[LU_diag[r]] + 1; cur_ptr < LU_colptr_trans[r + 1];
         cur_ptr++) {
      int nxt_r = LU_rowidx_trans[cur_ptr];
      LU_buf[nxt_r] = 0;
    }
  }
  //  if (!iam) printf(" -- flop: %lld %lld %lld %lld\n", flop1, idx1, flop2,
  //  idx2);
}
// void core_numfact_gpu() {
//   static bool graph_created = false;
//   static cudaGraph_t graph;
//   static cudaGraphExec_t instance;
//   static cudaStream_t stream;
//   int numThreadsPerBlock, numBlocks;
//   START()
//   numThreadsPerBlock = 32;
//   numBlocks = 80;
//   // sparseLU2<<<numBlocks, numThreadsPerBlock>>>(LU_data_gpu, LU_rowptr_gpu,
//   // LU_colidx_gpu, LU_diag_gpu, L_colptr_trans_gpu, L_rowidx_trans_gpu,
//   // L_trans_bt_gpu, leaf_size, core_n);

//   cudaDeviceSynchronize();

//   // gpuErrchk(cudaStreamCreate(&stream));
//   stream = cudaStreamDefault;
//   cudaMemset(LU_buf_gpu, 0, leaf_size * sizeof(double));

//   // if (!graph_created) {
//   // gpuErrchk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
//   // cudaGraphCreate(&graph, 0);
//   for (int r = 0; r < leaf_size; r++) {  // leaf_size
//     sparseLU_1<<<1, 1, 0, stream>>>(LU_data_gpu, LU_rowptr_gpu,
//     LU_colidx_gpu,
//                                     LU_diag_gpu, L_colptr_trans_gpu,
//                                     L_rowidx_trans_gpu, L_trans_bt_gpu,
//                                     leaf_size, core_n, r);
//     // cudaMemset(LU_buf_gpu, 0, leaf_size * sizeof(double));
//     sparseLU_2<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
//         LU_data_gpu, LU_rowptr_gpu, LU_colidx_gpu, LU_diag_gpu,
//         L_colptr_trans_gpu, L_rowidx_trans_gpu, L_trans_bt_gpu, LU_buf_gpu,
//         leaf_size, core_n, r);
//     sparseLU_3<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
//         LU_data_gpu, LU_rowptr_gpu, LU_colidx_gpu, LU_diag_gpu,
//         LU_colptr_trans_gpu, LU_rowidx_trans_gpu, LU_trans_map_gpu,
//         LU_map_gpu, LU_buf_gpu, leaf_size, core_n, r);
//     sparseLU_4<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
//         LU_data_gpu, LU_rowptr_gpu, LU_colidx_gpu, LU_diag_gpu,
//         L_colptr_trans_gpu, L_rowidx_trans_gpu, L_trans_bt_gpu, LU_buf_gpu,
//         leaf_size, core_n, r);
//   }

//   //   gpuErrchk(cudaStreamEndCapture(stream, &graph));
//   //   gpuErrchk(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
//   //   graph_created = true;
//   // }
//   // else {
//   //   cudaGraphLaunch(instance, stream);
//   //   cudaStreamSynchronize(stream);
//   // }

//   // for (int r = 0; r < leaf_size; r++) {
//   //   double tmp = LU_data[LU_diag[r]];
//   //   if (tmp < eps && tmp > -eps) {
//   //     if (tmp < 0)
//   //       tmp = -eps;
//   //     else
//   //       tmp = eps;
//   //   }
//   //   LU_data[LU_diag[r]] = tmp;

//   //   for(int cur_ptr = L_colptr_trans[r]; cur_ptr<L_colptr_trans[r+1];
//   //   cur_ptr++) { int nxt_r = L_rowidx_trans[cur_ptr];

//   //     int nxt_itr=L_trans_bt[cur_ptr];
//   //     double d = LU_data[nxt_itr]/tmp;
//   //     LU_data[nxt_itr]/=tmp;

//   //     for (int cur_itr = LU_diag[r]+1; cur_itr < LU_rowptr[r+1];
//   cur_itr++) {
//   //       int cur_c=LU_colidx[cur_itr];
//   //       while (LU_colidx[nxt_itr] < cur_c) { nxt_itr++;}
//   //       LU_data[nxt_itr] -= LU_data[cur_itr] * d;
//   //     }
//   //   }
//   // }

//   // gpuErrchk(cudaMemcpy(LU_data_gpu, LU_data, (LU_nnz) * sizeof(double),
//   //                      cudaMemcpyHostToDevice));
//   cudaDeviceSynchronize();
//   if ((!iam) && 0) cout << "\t" << iam << " step0-2 " << GET() << endl;
//   // gpuErrchk(cudaMemcpy(LU_data_gpu, LU_data, (LU_nnz) * sizeof(double),
//   // cudaMemcpyHostToDevice));

//   START()
//   numThreadsPerBlock = 256;
//   numBlocks =
//       ((core_n - leaf_size) + numThreadsPerBlock - 1) / numThreadsPerBlock;
//   updateLU<<<numBlocks, numThreadsPerBlock>>>(LU_data_gpu, LU_rowptr_gpu,
//                                               LU_colidx_gpu, LU_diag_gpu,
//                                               leaf_size, core_n);
//   cudaDeviceSynchronize();
//   gpuErrchk(cudaGetLastError());
//   if ((!iam) && 0) cout << "\t" << iam << " step0-3 " << GET() << endl;
// }
void core_trsm() {
  for (int r = 0; r < leaf_size; r++) {
    double tmp = 0;
    for (int ptr = LU_rowptr[r]; ptr < LU_diag[r]; ptr++) {
      int c = LU_colidx[ptr];
      tmp += LU_data[ptr] * _b[c];
      sparse_flop_trsm+=2;
    }
    _b[r] -= tmp;
  }
}
// void core_trsm_gpu() {
//   sparseTRSM_kernel<<<1, 32>>>(_b_gpu, LU_rowptr_gpu, LU_colidx_gpu,
//                                LU_diag_gpu, LU_data_gpu, L_colptr_trans_gpu,
//                                L_rowidx_trans_gpu, L_trans_bt_gpu,
//                                leaf_size);
//   cudaDeviceSynchronize();
// }
void core_update_b() {
  for (int r = leaf_size; r < core_n; r++) {
    double tmp = 0;
    for (int ptr = LU_rowptr[r]; ptr < LU_rowptr[r + 1]; ptr++) {
      int c = LU_colidx[ptr];
      tmp += LU_data[ptr] * _b[c];
    }
    _b[r] -= tmp;
  }
}
// void core_update_b_gpu() {
//   int numThreadsPerBlock = 256;
//   int numBlocks =
//       (core_n - leaf_size + numThreadsPerBlock - 1) / numThreadsPerBlock;
//   backwardSubstitution<<<1, 1>>>(_b_gpu, LU_rowptr_gpu, LU_colidx_gpu,
//                                  LU_data_gpu, leaf_size, core_n);
//   cudaDeviceSynchronize();
// }
void core_GEMM() {
  // #pragma omp parallel for num_threads(8)
  for (int r = leaf_size; r < core_n; r++) {
    int bias = r - (dense_row + 1) * leaf_size;
    for (int ptr1 = LU_rowptr[r]; ptr1 < LU_rowptr[r + 1]; ptr1++) {
      int r2 = LU_colidx[ptr1];
      double val = LU_data[ptr1];

      for (int ptr2 = LU_bias[r2]; ptr2 < LU_rowptr[r2 + 1]; ptr2++) {
        int c = LU_colidx[ptr2];
        double val2 = LU_data[ptr2];
        MM_buf[c * dense_row + bias] -= val * val2;
        sparse_flop_gemm+=2;
      }
    }
  }
}
// void core_GEMM_gpu() {
//   int numThreadsPerBlock = 256;
//   int numBlocks = (dense_row + numThreadsPerBlock - 1) / numThreadsPerBlock;
//   sparseSPMM_kernel2<<<numThreadsPerBlock, numBlocks>>>(
//       MM_buf_gpu, LU_rowptr_gpu, LU_colidx_gpu, LU_data_gpu, LU_bias_gpu,
//       leaf_size, core_n, dense_row);
// }
void core_update() {
  for (int j1 = leaf / 2; j1; j1 /= 2) {
    for (int j2 = leaf / 2; j2; j2 /= 2) {
      auto &e = LU[{j1, j2}];
      int r1 = block_start[j1], r2 = block_start[j1] + block_size[j1];
      int c1 = block_start[j2], c2 = block_start[j2] + block_size[j2];

      for (int r = r1; r < r2; r++) {
        for (int c = c1; c < c2; c++) {
          e.data[(r - r1) + (c - c1) * (r2 - r1)] =
              MM_buf[(r - leaf_size) + (c - leaf_size) * dense_row];
        }
      }
    }
  }
}
// void core_update_gpu() {
//   for (int j1 = leaf / 2; j1; j1 /= 2) {
//     for (int j2 = leaf / 2; j2; j2 /= 2) {
//       auto &e = LU[{j1, j2}];
//       int r1 = block_start[j1], r2 = block_start[j1] + block_size[j1];
//       int c1 = block_start[j2], c2 = block_start[j2] + block_size[j2];
//       dim3 blockSize(16, 16);
//       dim3 gridSize((c2 - c1 + blockSize.x - 1) / blockSize.x,
//                     (r2 - r1 + blockSize.y - 1) / blockSize.y);
//       updateData<<<gridSize, blockSize>>>(e.data_gpu, MM_buf_gpu, r1, r2, c1,
//                                           c2, dense_row, leaf_size);
//     }
//   }
// }
void core_run() {
  int numThreadsPerBlock, numBlocks;
  for (int i = 0; i < LU_nnz; i++)
    LU_data[i] = 0;
  for (int i = 0; i < core_nnz; i++) {
    LU_data[core_map[i]] = loc_val[i];
  }

  START()
  core_numfact_v2();
  double getrf_time=GET();
  if ((!iam) && 1)
    cout << "\t" << iam << " step1-1 (First level LU-2) : " << getrf_time << endl;

  START();
  core_trsm();
  double trsm_time=GET();
  // core_trsm_gpu();
  if ((!iam) && 1)
    cout << "\t" << iam << " step1-2 (First level b TRSM) : " << trsm_time << endl;

  START();
  core_update_b();
  // core_update_b_gpu();
  if ((!iam) && 1)
    cout << "\t" << iam << " step1-2 (First level b update)" << GET() << endl;

  START();
  // core_togpu();
  // b_togpu();
  if ((!iam) && 1)
    cout << "\t" << iam << " step1-* (First level communication) : " << GET()
         << endl;

  START()
  // gpuErrchk(cudaMemcpy(MM_buf_gpu, MM_buf,
  // dense_row*dense_row*sizeof(double), cudaMemcpyHostToDevice));
  // core_GEMM_gpu();
  core_GEMM();
  double gemm_time=GET();
  if ((!iam) && 1)
    cout << "\t" << iam << " step2-1 (First level GEMM) : " << gemm_time << endl;

  START()
  
  core_reduction();
  // core_update_gpu();
  core_update();
  if ((!iam) && 1)
    cout << "\t" << iam << " step2-2 (First level GEMM update) : " << GET()
         << endl;
  MPI_Barrier(MPI_COMM_WORLD);

  long long *sparse_flop_gemm_gather, *sparse_flop_getrf_gather, *sparse_flop_trsm_gather;
  double *getrf_time_gather, *gemm_time_gather, *trsm_time_gather;

  sparse_flop_gemm_gather=(long long*)malloc(sizeof(long long) * np );
  sparse_flop_getrf_gather=(long long*)malloc(sizeof(long long) * np );
  sparse_flop_trsm_gather=(long long*)malloc(sizeof(long long) * np );
  gemm_time_gather=(double*)malloc(sizeof(double) * np );
  getrf_time_gather=(double*)malloc(sizeof(double) * np );
  trsm_time_gather=(double*)malloc(sizeof(double) * np );


  MPI_Gather(&sparse_flop_getrf, 1, MPI_INT64_T, sparse_flop_getrf_gather, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
  MPI_Gather(&sparse_flop_trsm, 1, MPI_INT64_T, sparse_flop_trsm_gather, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
  MPI_Gather(&sparse_flop_gemm, 1, MPI_INT64_T, sparse_flop_gemm_gather, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

  MPI_Gather(&getrf_time, 1, MPI_DOUBLE, getrf_time_gather, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&trsm_time, 1, MPI_DOUBLE, trsm_time_gather, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&gemm_time, 1, MPI_DOUBLE, gemm_time_gather, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  // double tmp=(sparse_flop_gemm+sparse_flop_getrf+sparse_flop_trsm);
  if ((!iam) && 1) {
      // Get the current date and time to create the file name
      time_t now = time(NULL);
      struct tm *t = localtime(&now);
      char filename[100];
      snprintf(filename, sizeof(filename), "sparse_log_%04d-%02d-%02d_%02d-%02d-%02d.txt",
              t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
              t->tm_hour, t->tm_min, t->tm_sec);

      // Open the file for writing
      FILE *file = fopen(filename, "w");
      if (file == NULL) {
        perror("Unable to open file");
      }

      fprintf(file, "SGETRF\n");
      for (int i = 0; i < np; i++) {
          fprintf(file, " %d %lf\n", sparse_flop_getrf_gather[i], getrf_time_gather[i]);
      }

      fprintf(file, "STRSM\n");
      for (int i = 0; i < np; i++) {
          fprintf(file, " %d %lf\n", sparse_flop_trsm_gather[i], trsm_time_gather[i]);
      }

      fprintf(file, "SGEMM\n");
      for (int i = 0; i < np; i++) {
          fprintf(file, " %d %lf\n", sparse_flop_gemm_gather[i], gemm_time_gather[i]);
      }

      // Close the file
      fclose(file);
  }

  /*
  for (int r = 0; r < leaf_size; r++) {
    double tmp = 0;
    for (int ptr = LU_rowptr_gpu[r]; ptr < LU_diag_gpu[r]; ptr++) {
      int c = LU_colidx_gpu[ptr];
      tmp += LU_data_gpu[ptr] * _b[c];
    }
    _b[r] -= tmp;
  }

  for (int r = leaf_size; r < core_n; r++) {
    int j1 = which_block[r];
    for (int k = 0; k < core_n; k++) LU_itr[k] = LU_bias[k];
    for (int j2 = leaf / 2; j2; j2 /= 2) {
      auto &e = LU[{j1, j2}];
      int bias = (r - block_start[j1]) * e.m - block_start[j2];
      for (int ptr1 = LU_rowptr[r]; ptr1 < LU_rowptr[r + 1]; ptr1++) {
        int r2 = LU_colidx[ptr1];

        int ptr2;
        for (ptr2 = LU_itr[r2]; ptr2 < LU_rowptr[r2 + 1]; ptr2++) {
          int c = LU_colidx[ptr2];
          if(c<block_start[j2]) continue;
          if (c >= block_start[j2] + block_size[j2]) break;

          e.data[bias + c] -= LU_data[ptr1] * LU_data[ptr2];
        }
        LU_itr[r2] = ptr2;
      }
    }
  }

  for (int r = leaf_size; r < core_n; r++) {
    for (int ptr = LU_rowptr[r]; ptr < LU_rowptr[r + 1]; ptr++) {
      int c = LU_colidx[ptr];
      _b[r] -= LU_data[ptr] * _b[c];
    }
  }
  */
}


