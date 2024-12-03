#include <vector>
#include <map>
#include "mpi.h"

#include "SnuMat.h"
#include "kernel.h"

using namespace std;

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void vectorAdd(double *A, const double *B, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) {
    A[i] = A[i] + B[i];
  }
}

void SnuMat::reduction(int block_num) { // Todo change to MPI_REDUCE & non blocking

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
void SnuMat::reduction_gpu(int block_num) { // Todo change to MPI_REDUCE & non blocking
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
void SnuMat::scatter_b(int block_num) { // Todo change to MPI_REDUCE & non blocking
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
void SnuMat::scatter_b_gpu(int block_num) { // Todo change to MPI_REDUCE & non blocking
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
void SnuMat::gather_data_b() {
  for (int i = num_block; i >= 1; i--) {
    if (!(who[i])) {
      memcpy(perm_b + old_block_start[i], b[i].data,
             block_size[i] * sizeof(double));
    } else {
      MPI_Recv(perm_b + old_block_start[i], block_size[i], MPI_DOUBLE, who[i],
              0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

void SnuMat::return_data_b() {
  for (auto &i : my_block) {
    MPI_Send(b[i].data, b[i].n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
}
void SnuMat::b_togpu() {
  gpuErrchk(cudaMemcpy(this->_b_gpu, this->_b, this->local_b_rows * sizeof(double),
                       cudaMemcpyHostToDevice));
}
void SnuMat::b_tocpu() {
  gpuErrchk(cudaMemcpy(this->_b, this->_b_gpu, this->local_b_rows * sizeof(double),
                       cudaMemcpyDeviceToHost));
}
void SnuMat::solve(double *x) {
  core_run();
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
  
  // printf("%d: ", iam); for(int i=0; i<local_b_rows; i++) printf("%lf ", _b[i]); printf("\n");
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

  if (offlvl >= 0 && (!(iam & 1)))
    b_tocpu();
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

  {
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

    for (int r = leaf_size - 1; r >= 0; r--) {
      for (int ptr = LU_diag[r] + 1; ptr < LU_bias[r]; ptr++) {
        int c = LU_colidx[ptr];
        _b[r] -= LU_data[ptr] * _b[c];
      }
      _b[r] /= LU_data[LU_diag[r]];
    }
  }

  if (!iam) {
    gather_data_b();
    for (int i = 0; i < n; i++)
      x[i] = perm_b[order[i]];
  } else {
    return_data_b();
  }
}