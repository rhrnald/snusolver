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

void SnuMat::distribute_structure() {
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
}