#ifndef SNUSOLVER_SNUMAT
#define SNUSOLVER_SNUMAT

#include <iostream>
#include <cstdio>
#include <vector>
#include <map>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "mpi.h"

#include "timer.h"
#include "param.h"
#include "matrix.h"
using namespace std;

class SnuMat {
public:
  //basic
  int np, iam;
  cublasHandle_t handle;
  cusolverDnHandle_t cusolverHandle;
  int _who(int block);
  void make_who();


  //matrix info
  int n, nnz;
  int num_block, max_level;

  //distribute
  int who[NP + NP];
  int* order, *sizes;
  vector<int> my_block, all_parents;
  vector<int> my_block_level[LEVEL + 1];
  int* old_block_start, *block_start, *block_size;
  map<pair<int, int>, dense_matrix> LU;
  MPI_Request *request;
  int level[NP + NP];
  int *coo_r, *coo_c;
  int *loc_r, *loc_c;
  double *coo_val;
  coo_matrix *A;
  map<pair<int, int>, int> Amap;
  int rs_cur, rs_idx, max_nnz;
  int *merge_start, *merge_size;
  int *send_order;
  int *perm_map;

  int mat_cnt, loc_nnz;

  coo_matrix L[NP + NP][LEVEL + 1];
  coo_matrix U[NP + NP][LEVEL + 1];

  coo_matrix& grid(int i, int j) {
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
  void dfs_block_order(vector<int> &v, int cur, int parts);csr_matrix permutate(csr_matrix A, int *order);
  void calculate_block(int *sizes, int np);
  void clearGrid(int i, int j);
  int setGrid(int i, int j, int bias);
  int addElement(int i, int j, int r, int c);
  void construct(csr_matrix A);
  void self_submatrix(int i, int j);
  void send_submatrix(int proc, int i, int j);
  void receive_submatrix(int i, int j);
  void set_LU(int i, int j) ;
  void set_all_LU(int i);
  void send_data_merge_async(int i, MPI_Request *t);
  void send_data_a();
  void get_data_a_async(int e);
  void get_data_a();
  void send_data_b();
  void get_data_b();
  void malloc_LU(int i, int j, int lvl);
  void free_LU(int i, int j, int lvl);
  void clear_LU(int i, int j);
  void malloc_all_LU();
  void free_all_LU();
  void malloc_all_b();
  void free_all_b();
  void clear_all_LU();
  


  //sparse
  int LU_nnz, core_nnz, core_n, core_m;
  double *LU_data, *LU_data_gpu, *LU_buf, *LU_buf_gpu;
  int *LU_buf_int;
  int *LU_colidx, *LU_rowptr, *LU_diag, *LU_bias;
  int *LU_colptr_trans, *LU_rowidx_trans, *LU_diag_trans;
  int *LU_map, *LU_trans_map;
  int *LU_trans_map_gpu;
  int *LU_colidx_gpu, *LU_rowptr_gpu, *LU_diag_gpu, *LU_bias_gpu;
  int *LU_colptr_trans_gpu, *LU_rowidx_trans_gpu, *LU_diag_trans_gpu;
  int *LU_map_gpu;
  int *L_colptr_trans, *L_rowidx_trans, *L_trans_bt;
  int *L_colptr_trans_gpu, *L_rowidx_trans_gpu, *L_trans_bt_gpu;
  double *core_data;
  int *core_rowptr, *core_colidx, *core_map;
  int leaf, leaf_size, local_b_rows, dense_row;
  double *_b, *_b_gpu;
  double *perm_b;
  double *loc_val;
  map<int, dense_matrix> b;
  double *MM_buf, *MM_buf2, *MM_buf_gpu, *gpu_data_buf;
  void core_togpu();
  void core_reduction();
  void core_symbfact();
  void core_numfact_v1();
  void core_numfact_v2();
  void core_numfact_v3();
  void core_trsm();
  void core_update_b();
  void core_GEMM();
  void core_update();

  void core_run();
  void core_preprocess();


  //solve
  void b_togpu();
  void b_tocpu();
  void reduction(int block_num);
  void reduction_gpu(int block_num);
  void scatter_b(int block_num);
  void scatter_b_gpu(int block_num);
  void gather_data_b();
  void return_data_b();

  //main procedure
  SnuMat(csr_matrix A_csr, double *b, cublasHandle_t handle, cusolverDnHandle_t cusolverHandle);
  void construct_structure(csr_matrix A_csr, int *sizes, int *order, double *b);
  void distribute_structure();
  void malloc_matrix();
  void distribute_data();
  void solve(double *x);
};
#endif