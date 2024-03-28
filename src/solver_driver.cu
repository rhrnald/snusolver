#include <mpi.h>
#include <omp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <vector>

#include "kernel.h"
#include "kernel_gpu.h"
#include "mkl.h"
#include "parmetis.h"
#include "snusolver.h"
#include "solver_driver.h"

using namespace std;

#include <chrono>

std::chrono::time_point<std::chrono::system_clock> s, e;

#define START() s = std::chrono::system_clock::now();
#define END() e = std::chrono::system_clock::now();
#define GET()                                                                  \
  (std::chrono::duration_cast<std::chrono::duration<float>>(                   \
       (e = std::chrono::system_clock::now()) - s)                             \
       .count())

static int ngpus, omp_mpi_level, max_level;
// static int iam;
static int nnz, cols, rows, num_block, local_b_rows;
static MPI_Comm parent, metis_comm = MPI_COMM_WORLD;
static MPI_Request *request;

static const int LEVEL = 10;
static const int PARTS = 1 << LEVEL;

static const double eps = 1e-15;
static int offlvl = 10;

static int who[PARTS + PARTS];
static int level[PARTS + PARTS];
static vector<int> my_block, my_block_level[LEVEL + 1];
static int *merge_start, *merge_size;
static vector<int> all_parents;
static int *block_start, *block_size, *which_block;

static int mat_cnt, loc_nnz, max_nnz, leaf, leaf_size;

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

void print_gpu_mem() {
  size_t free_byte, total_byte;
  cudaMemGetInfo(&free_byte, &total_byte);
  std::cout << iam << "  " << free_byte / 1.0e9 << " Gbytes / "
            << total_byte / 1.0e9 << " Gbytes\n";
}

void dense_matrix::toCPU() {
  cudaMemcpy(data, data_gpu, n * m * sizeof(double), cudaMemcpyDeviceToHost);
}
void dense_matrix::toGPU() {
  // if (able)
  cudaMemcpy(data_gpu, data, n * m * sizeof(double), cudaMemcpyHostToDevice);
}
static int _who(int block) {
  while (block >= ngpus + ngpus)
    block /= 2;
  while (block < ngpus)
    block = block + block + 1;
  return ngpus + ngpus - block - 1;
}

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

static void malloc_all_LU() {
  for (auto &i : all_parents) {
    if (i >= ngpus)
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
    if (i >= ngpus)
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
    if (i >= ngpus)
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
  if (i >= ngpus)
    return;
  set_LU(i, i);
  for (int j = i / 2; j >= 1; j /= 2) {
    set_LU(i, j);
    set_LU(j, i);
  }
  //
}

static int rs_idx, rs_cur;
static void receive_submatrix(int i, int j) {
  int nnz;
  MPI_Recv(&nnz, 1, MPI_INT, 0, 0, parent, MPI_STATUS_IGNORE);

  int b = rs_cur;
  rs_cur += nnz;
  max_nnz = max(max_nnz, nnz);

  A[rs_idx] = {block_size[i], block_size[j], nnz,
               loc_r + b,     loc_c + b,     loc_val + b};
  Amap[{i, j}] = rs_idx;
  rs_idx++;

  MPI_Recv(loc_r + b, nnz, MPI_INT, 0, 0, parent, MPI_STATUS_IGNORE);
  MPI_Recv(loc_c + b, nnz, MPI_INT, 0, 0, parent, MPI_STATUS_IGNORE);

  for (int idx = 0; idx < nnz; idx++)
    loc_r[b + idx] -= block_start[i];
  for (int idx = 0; idx < nnz; idx++)
    loc_c[b + idx] -= block_start[j];
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
void get_data_a(int e) {
  MPI_Recv(loc_val + merge_start[e], merge_size[e], MPI_DOUBLE, 0, e, parent,
           MPI_STATUS_IGNORE);
}

void get_data_a_async(int e) {
  MPI_Irecv(loc_val + merge_start[e], merge_size[e], MPI_DOUBLE, 0, e, parent,
            &(request[e]));
}
void get_data_b() {
  std::fill(_b, _b + local_b_rows, 0.0);
  for (auto &i : my_block) {
    MPI_Recv(b[i].data, b[i].n, MPI_DOUBLE, 0, 0, parent, MPI_STATUS_IGNORE);
  }
}
void return_data_b() {
  for (auto &i : my_block) {
    MPI_Send(b[i].data, b[i].n, MPI_DOUBLE, 0, 0, parent);
  }
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

        // double *tmp;
        // gpuErrchk(cudaMalloc(&tmp, sizeof(double) * cnt));
        MPI_Recv(MM_buf_gpu, cnt * sizeof(double), MPI_BYTE, src,
                 j1 * num_block + j2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // gpuErrchk(cudaMemcpy(MM_buf_gpu, M.data, cnt * sizeof(double),
        //                      cudaMemcpyHostToDevice));
        vectorAdd<<<gridSize, blockSize>>>(M.data_gpu, MM_buf_gpu, cnt);
        cudaDeviceSynchronize();
        // cudaFree(tmp);
      }

      // int cnt = b[j1].n;
      // int blockSize = 256;
      // int gridSize = (cnt + blockSize - 1) / blockSize;

      // MPI_Recv(b[j1].data, cnt, MPI_DOUBLE, src, j1, MPI_COMM_WORLD,
      //          MPI_STATUS_IGNORE);
      // gpuErrchk(cudaMemcpy(gpu_data_buf, b[j1].data, cnt * sizeof(double),
      //                      cudaMemcpyHostToDevice));
      // vectorAdd<<<gridSize, blockSize>>>(b[j1].data_gpu, gpu_data_buf, cnt);
      // cudaDeviceSynchronize();
    }
    int cnt = core_n - block_start[block_num / 2];
    int blockSize = 256;
    int gridSize = (cnt + blockSize - 1) / blockSize;

    MPI_Recv(gpu_data_buf, cnt, MPI_DOUBLE, src, block_num / 2, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    // gpuErrchk(cudaMemcpy(gpu_data_buf, b[block_num/2].data, cnt *
    // sizeof(double),
    //                       cudaMemcpyHostToDevice));
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
        // gpuErrchk(cudaMemcpy(M.data, M.data_gpu, cnt * sizeof(double),
        //                      cudaMemcpyDeviceToHost));
        MPI_Send(M.data_gpu, cnt * sizeof(double), MPI_BYTE, dst,
                 j1 * num_block + j2, MPI_COMM_WORLD);
      }

      // int cnt = b[j1].n;
      // double *tmp = (double *) malloc(sizeof(double) * cnt);
      // gpuErrchk(cudaMemcpy(tmp, b[j1].data_gpu, cnt * sizeof(double),
      //                      cudaMemcpyDeviceToHost));
      // MPI_Send(tmp, cnt, MPI_DOUBLE, dst, j1, MPI_COMM_WORLD);
      // free(tmp);
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
  which_block = (int *)malloc(sizeof(int) * core_n);

  for (auto &i : all_parents) {
    for (int j = block_start[i]; j < block_start[i] + block_size[i]; j++)
      which_block[j] = i;
  }

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
//   if ((!iam) && 1) cout << "\t" << iam << " step0-2 " << GET() << endl;
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
//   if ((!iam) && 1) cout << "\t" << iam << " step0-3 " << GET() << endl;
// }
void core_trsm() {
  for (int r = 0; r < leaf_size; r++) {
    double tmp = 0;
    for (int ptr = LU_rowptr[r]; ptr < LU_diag[r]; ptr++) {
      int c = LU_colidx[ptr];
      tmp += LU_data[ptr] * _b[c];
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
  if ((!iam) && 1)
    cout << "\t" << iam << " step1-1 (First level LU-2) : " << GET() << endl;

  START();
  core_trsm();
  // core_trsm_gpu();
  if ((!iam) && 1)
    cout << "\t" << iam << " step1-2 (First level b TRSM) : " << GET() << endl;

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
  core_reduction();
  if ((!iam) && 1)
    cout << "\t" << iam << " step2-1 (First level GEMM) : " << GET() << endl;

  START()
  // core_update_gpu();
  core_update();
  if ((!iam) && 1)
    cout << "\t" << iam << " step2-2 (First level GEMM update) : " << GET()
         << endl;

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

static void get_structure() {
  MPI_Bcast(&nnz, 1, MPI_INT, 0, parent);
  MPI_Bcast(&cols, 1, MPI_INT, 0, parent);
  MPI_Bcast(&rows, 1, MPI_INT, 0, parent);

  MPI_Bcast(&num_block, 1, MPI_INT, 0, parent);

  block_start = (int *)malloc(sizeof(int) * (num_block + 1));
  block_size = (int *)malloc(sizeof(int) * (num_block + 1));
  request = (MPI_Request *)malloc(sizeof(MPI_Request) * (num_block + 1));

  MPI_Bcast(block_start + 1, num_block, MPI_INT, 0, parent);
  MPI_Bcast(block_size + 1, num_block, MPI_INT, 0, parent);

  who[0] = 0;
  for (int i = 1; i <= num_block; i++)
    who[i] = _who(i);
  for (int i = 0; i < PARTS + PARTS; i++)
    level[i] = 0;
  for (int e = 1; e <= num_block; e++) {
    for (int i = e; i > 1; i /= 2)
      level[e]++;
  }

  for (int i = num_block; i >= 1; i--)
    if (who[i] == iam)
      my_block.push_back(i);

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

  MPI_Scatter(nullptr, 0, 0, &mat_cnt, 1, MPI_INT, 0, parent);
  MPI_Scatter(nullptr, 0, 0, &loc_nnz, 1, MPI_INT, 0, parent);

  loc_r = (int *)malloc(sizeof(int) * loc_nnz);
  loc_c = (int *)malloc(sizeof(int) * loc_nnz);
  loc_val = (double *)malloc(sizeof(double) * loc_nnz);

  A = (coo_matrix *)malloc(sizeof(coo_matrix) * mat_cnt);

  max_nnz = 0;
  rs_idx = 0;
  rs_cur = 0;

  merge_start = (int *)malloc(sizeof(int) * (num_block + 1));
  merge_size = (int *)malloc(sizeof(int) * (num_block + 1));

  for (auto &i : my_block) {
    merge_start[i] = rs_cur;
    receive_submatrix(i, i);
    for (int ii = i / 2; ii; ii /= 2) {
      receive_submatrix(i, ii);
      receive_submatrix(ii, i);
    }
    merge_size[i] = rs_cur - merge_start[i];
  }

  leaf = ngpus + ngpus - 1 - iam;
  leaf_size = block_size[leaf];

  local_b_rows = 0;
  for (auto &i : all_parents)
    local_b_rows += block_size[i];
  dense_row = local_b_rows - leaf_size;

  malloc_all_LU();
  malloc_all_b();
  core_preprocess();
}

// __global__ void kernel4_1(double *_b, int *LU_rowptr, int *LU_colidx,
//                           double *LU_data, int *LU_bias, int *LU_diag,
//                           int leaf_size) {
//   int r = blockIdx.x * blockDim.x + threadIdx.x;
//   if (r >= leaf_size) return;
//   // for (int r = 0; r < leaf_size; r++) {
//   for (int ptr = LU_rowptr[r + 1] - 1; ptr >= LU_bias[r]; ptr--) {
//     int c = LU_colidx[ptr];
//     _b[r] -= LU_data[ptr] * _b[c];
//   }
//   //}
// }
// __global__ void kernel4_2(double *_b, int *LU_rowptr, int *LU_colidx,
//                           double *LU_data, int *LU_bias, int *LU_diag,
//                           int *LU_colptr_trans, int *LU_rowidx_trans,
//                           int *LU_diag_trans, int *LU_trans_map,
//                           int leaf_size) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int N = gridDim.x * blockDim.x;
//   for (int r = leaf_size - 1; r >= 0; r--) {
//     __syncthreads();
//     if (idx == 0) { _b[r] /= LU_data[LU_diag[r]]; }
//     __syncthreads();
//     double d = _b[r];

//     for (int ptr = LU_colptr_trans[r] + idx; ptr < LU_diag_trans[r]; ptr +=
//     N) {
//       int c = LU_rowidx_trans[ptr];
//       int p = LU_trans_map[ptr];
//       _b[c] -= LU_data[p] * _b[r];
//     }
//   }
// }

static void solve() {
  get_data_b();
  // for(auto &i : my_block) get_data_a(i);
  // for(auto &i : my_block) set_all_LU(i);
  for (auto &i : my_block)
    get_data_a_async(i);
  for (auto &i : my_block) {
    MPI_Wait(&request[i], MPI_STATUS_IGNORE);
    set_all_LU(i);
  }

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

  return_data_b();
}
static void finalize() {
  free_all_LU();
  free_all_b();

  free(block_start);
  free(block_size);
  free(which_block);
  free(loc_r);
  free(loc_c);
  free(loc_val);
  free(A);

  free(core_rowptr);
  free(core_colidx);
  free(core_map);
  free(core_data);

  free(LU_rowptr);
  free(LU_colidx);
  free(LU_diag);
  free(LU_data);
  free(LU_bias);

  free(MM_buf);
  free(MM_buf2);
  free(LU_buf);
  free(LU_buf_int);

  free(L_colptr_trans);
  free(L_rowidx_trans);
  free(L_trans_bt);
  free(LU_colptr_trans);
  free(LU_rowidx_trans);
  free(LU_trans_map);
  free(LU_map);
  free(LU_diag_trans);

  free(merge_start);
  free(merge_size);
  free(request);
  my_block.clear();
  for (int i = 0; i <= LEVEL; i++)
    my_block_level[i].clear();
  all_parents.clear();
  Amap.clear();
  LU.clear();
  b.clear();

  cublasDestroy(handle);
  cusolverDnDestroy(cusolverHandle);
}

int main_solver() {
  // MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &omp_mpi_level);
  MPI_Comm_get_parent(&parent);

  MPI_Comm_rank(metis_comm, &iam);
  MPI_Comm_size(metis_comm, &ngpus);
  max_level = 0;

  for (int i = ngpus; i > 1; i /= 2)
    max_level++;
  // MPI_Get_processor_name(processor_name, &name_len);

  // printf("Rank %d of %d on node %s\n", iam, ngpus, processor_name);
  //  mkl_set_num_threads(32/ngpus);
  gpuErrchk(cudaSetDevice(iam % 4));
  cublasCreate(&handle);
  cusolverDnCreate(&cusolverHandle);
  get_structure();

  // int max_block_size = 0;
  // for (int i = (num_block + 1) / 2; i >= 1; i--) {
  //   max_block_size = max(max_block_size, block_size[i]);
  // }
  // max_block_size = max(max_block_size * max_block_size, max_nnz);
  if (offlvl >= 0 && (!(iam & 1))) {
    gpuErrchk(cudaMalloc((void **)&gpu_row_buf, max_nnz * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&gpu_col_buf, max_nnz * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&gpu_data_buf, max_nnz * sizeof(double)));
  }

  SOLVER_COMMAND command;
  while (true) {
    clear_all_LU();
    MPI_Bcast(&command, sizeof(SOLVER_COMMAND), MPI_BYTE, 0, parent);
    if (command == SOLVER_RUN)
      solve();
    else if (command == SOLVER_RESET) {
      finalize();
      return 1;
    } else {
      finalize();
      return 0;
    }
  }
}
