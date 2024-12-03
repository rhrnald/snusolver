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
void SnuMat::distribute_data() {
  if (!iam) {
    send_data_b();
    send_data_a();
  } else {
    get_data_b();
    get_data_a();
  }
}
