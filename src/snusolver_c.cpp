#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <fstream>
#include <iostream>

#include "mpi.h"
#include "snusolver.h"

using namespace std;
namespace py = pybind11;

////////////////////////////////////////////////////////////
/////                                                   ////
/////                 Static Variables                  ////
/////                                                   ////
////////////////////////////////////////////////////////////

vector<int> tmp1, tmp2;

static MPI_Comm childs;
static const int HLEVEL = 9;
static const int HPARTS = 1 << HLEVEL;
static int ngpus = HPARTS;

static const int LEVEL = HLEVEL + HLEVEL;
static const int PARTS = 1 << LEVEL;

static int *order, *sizes;

static int rows, cols, nnz;
static int *csr_rowptr, *csr_colidx;
static double *csr_data;
static int *coo_r, *coo_c, *perm_map;
static double *coo_val;

static int num_block;
static int *block_start, *block_size, *merge_size, *send_order;

static int *vtxdist, *loc_rowss, *displs, *loc_nnzs;
static coo_matrix L[PARTS + PARTS][LEVEL + 1];
static coo_matrix U[PARTS + PARTS][LEVEL + 1]; // grid[i][j] = A[i][i>>j];

static int who[PARTS + PARTS];

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

int test_first;
void test(int ngpus) {
  MPI_Info info;
  MPI_Info_create(&info);
  // MPI_Info_set(info, "hostfile", "hostfile.txt");
  // MPI_Info_set(info, "host", "a0:65,a1:33,a2:33,a3:33");
  MPI_Info_set(info, "npernode", "32");

  // char *argv[]={"--openacc-profiling", "off", "solver_driver", nullptr};
  MPI_Comm_spawn("solver_driver", MPI_ARGV_NULL, ngpus, info, 0, MPI_COMM_SELF,
                 &childs, MPI_ERRCODES_IGNORE);

  printf("!\n");
  fflush(stdout);
}

////////////////////////////////////////////////////////////
/////                                                   ////
/////                       Util                        ////
/////                                                   ////
////////////////////////////////////////////////////////////
#include <chrono>
static vector<std::chrono::time_point<std::chrono::system_clock>> start_time;

#define PRINT
#ifdef PRINT
#define START(s)                                                               \
  {                                                                            \
    for (int i = 0; i < (int)(start_time.size()); i++)                         \
      printf("\t");                                                            \
    std::cout << s << " start!" << endl;                                       \
    start_time.push_back(std::chrono::system_clock::now());                    \
  }
#define END(s)                                                                 \
  {                                                                            \
    auto st = start_time.back();                                               \
    start_time.pop_back();                                                     \
    std::chrono::time_point<std::chrono::system_clock> end_time =              \
        std::chrono::system_clock::now();                                      \
    for (int i = 0; i < (int)(start_time.size()); i++)                         \
      printf("\t");                                                            \
    std::cout << s << " done! ("                                               \
              << std::chrono::duration_cast<std::chrono::duration<float>>(     \
                     end_time - st)                                            \
                     .count()                                                  \
              << " sec)" << std::endl;                                         \
  }
#else
#define START(s)                                                               \
  {}
#define END(s)                                                                 \
  {}
#endif

static void dump(const char *filename, void *data, int len) {
  ofstream ofile;
  ofile.open(filename, std::ofstream::out | std::ofstream::binary);
  ofile.write(reinterpret_cast<char *>(data), len);
  ofile.close();
}

static int which(int block) {
  while (block >= ngpus + ngpus)
    block /= 2;
  while (block < ngpus)
    block = block + block + 1;
  return ngpus + ngpus - block - 1;
}

////////////////////////////////////////////////////////////
/////                                                   ////
/////               Function Definitions                ////
/////                                                   ////
////////////////////////////////////////////////////////////
void initialize(int _ngpus);
void analyze(py::object A);
void spawn_solver_processes();
vector<double> solve(py::array_t<double> Adata_py, py::array_t<double> b_py);
vector<double> _solve(double *data, py::array_t<double> b_py);
vector<double> solve_small(py::object A, py::array_t<double> b_py);
void reset_solver_processes();
void finalize_solver_processes();

static void spawn_parmetis_processes(int np, int nnz, int rows);
static void finalize_parmetis_processes();
static csc_matrix addTransposeEliminateDiag(csc_matrix A, int *buf1, int *buf2);
static void call_parmetis(csc_matrix A, int parts, int *order, int *sizes);
static csc_matrix permutate(csc_matrix A, int *order, int *buf1, int *buf2);
static void calculate_block(int *sizes, int parts);
static csc_matrix getSubmatrix(csc_matrix A, int i, int j, int parts, int *buf1,
                               int *buf2);
static void construct(csc_matrix A);
static void fill(int *sizes, int *subsizes, int head, int parts);

PYBIND11_MODULE(snusolver_c, m) {
  m.def("initialize", &initialize, "");
  m.def("analyze", &analyze, "");
  // m.def("spawn_solver_processes", &spawn_solver_processes, "");
  m.def("solve", &solve, "");
  m.def("solve_small", &solve_small, "");
  m.def("reset_solver_processes", &reset_solver_processes, "");
  m.def("finalize_solver_processes", &finalize_solver_processes, "");
  m.def("test", &test, "");
}

////////////////////////////////////////////////////////////
/////                                                   ////
/////                 Main Modules                      ////
/////                                                   ////
////////////////////////////////////////////////////////////

void initialize(int _ngpus) {
  START("intialize")
  ngpus = _ngpus;
  MPI_Init(nullptr, nullptr);

  MPI_Comm_spawn("driver", MPI_ARGV_NULL, _ngpus, MPI_INFO_NULL, 0,
                 MPI_COMM_SELF, &childs, MPI_ERRCODES_IGNORE);

  END("intialize")
}
// Partition A into parts*parts=1024 blocks.
// Permutation and Construct structure (Allocate memory and map)
void analyze(py::object A) {
  START("analyze")
  nnz = py::int_(A.attr("nnz"));
  rows = py::cast<int>(py::cast<py::tuple>(A.attr("shape"))[0]);
  cols = py::cast<int>(py::cast<py::tuple>(A.attr("shape"))[1]);
  int *rowidx = static_cast<int *>(
      py::cast<py::array_t<int>>(A.attr("indices")).request().ptr);
  int *colptr = static_cast<int *>(
      py::cast<py::array_t<int>>(A.attr("indptr")).request().ptr);
  csc_matrix PA;
  int parts = ngpus;

  csr_rowptr = (int *)malloc(sizeof(int) * (rows + 1));
  csr_colidx = (int *)malloc(sizeof(int) * (nnz));
  csr_data = (double *)malloc(sizeof(double) * (nnz));

  for (int i = 0; i < rows + 1; i++)
    csr_rowptr[i] = colptr[i];
  for (int i = 0; i < nnz; i++)
    csr_colidx[i] = rowidx[i];

  coo_r = (int *)malloc(sizeof(int) * nnz);
  coo_c = (int *)malloc(sizeof(int) * nnz);
  coo_val = (double *)malloc(sizeof(double) * nnz);
  perm_map = (int *)malloc(sizeof(int) * nnz);
  order = (int *)malloc(sizeof(int) * (rows));
  sizes = (int *)malloc(sizeof(int) * (parts * parts * 2 - 1));
  block_start = (int *)malloc(sizeof(int) * (parts * parts * 2));
  block_size = (int *)malloc(sizeof(int) * (parts * parts * 2));
  merge_size = (int *)malloc(sizeof(int) * (parts * parts * 2));
  send_order = (int *)malloc(sizeof(int) * (parts * parts * 2));

  int *buf1 = (int *)malloc(sizeof(int) * (rows + 1));
  int *buf2 = (int *)malloc(sizeof(int) * (nnz + nnz));
  int *buf3 = (int *)malloc(sizeof(int) * (rows + 1));
  int *buf4 = (int *)malloc(sizeof(int) * nnz);
  int *buf5 = (int *)malloc(sizeof(int) * (rows + 1));
  int *buf6 = (int *)malloc(sizeof(int) * (nnz + nnz));

  int *subsizes = (int *)malloc(sizeof(int) * (parts + parts - 1));
  int *newsizes = (int *)malloc(sizeof(int) * (parts * parts * 2 - 1));
  int *suborder = (int *)malloc(sizeof(int) * rows);
  int *neworder = (int *)malloc(sizeof(int) * rows);

  spawn_parmetis_processes(parts, nnz, rows);

  csc_matrix A_csc = {rows, cols, nnz, rowidx, colptr, nullptr};
  csc_matrix AAT = addTransposeEliminateDiag(A_csc, buf1, buf2);

  START("parmetis")
  call_parmetis(AAT, parts, order, sizes);
  END("parmetis")

  /*
  START("permutate")
  PA=permutate(A_csc, order, buf1, buf2);
  END("permutate")

  START("recursion")

  calculate_block(sizes, parts);

  for(int i=0; i<cols; i++) neworder[i]=i;
  fill(sizes, sizes, 1, parts);
  for(int i=parts+parts-1; i>=parts; i--) {
    csc_matrix tmp = getSubmatrix(PA, i, i, parts, buf3, buf4);
    csc_matrix tmp2 = addTransposeEliminateDiag(tmp,buf5,buf6);
    call_parmetis(tmp2, parts, suborder, subsizes);
    fill(sizes, subsizes, i, parts);
    int loc_cols=block_size[i];
    int bias=block_start[i];
    for(int ii=0; ii<loc_cols; ii++) neworder[ii+bias]=suborder[ii]+bias;
  }
  END("recursion")

  for(int i=0; i<rows; i++) order[i]=neworder[order[i]];*/

  finalize_parmetis_processes();

  free(neworder);
  free(suborder);
  free(newsizes);
  free(subsizes);

  free(buf6);
  free(buf5);
  free(buf4);
  free(buf3);
  END("analyze")

  START("construction");
  // parts=PARTS;

  calculate_block(sizes, parts);
  PA = permutate(A_csc, order, buf1, buf2);
  construct(PA);

  free(buf2);
  free(buf1);
  END("construction");

  spawn_solver_processes();
}

void send_submatrix(int proc, int i, int j) {
  int loc_nnz = grid(i, j).nnz;

  MPI_Send(&loc_nnz, 1, MPI_INT, proc, 0, childs);
  MPI_Send(grid(i, j).row, loc_nnz, MPI_INT, proc, 0, childs);
  MPI_Send(grid(i, j).col, loc_nnz, MPI_INT, proc, 0, childs);
}

static void send_data(int p, int i, int j) {
  MPI_Send(grid(i, j).data, grid(i, j).nnz, MPI_DOUBLE, p, 0, childs);
}
static void send_data_merge(int i) {
  MPI_Send(grid(i, i).data, merge_size[i], MPI_DOUBLE, who[i], i, childs);
}
static void send_data_merge_async(int i, MPI_Request *t) {
  MPI_Isend(grid(i, i).data, merge_size[i], MPI_DOUBLE, who[i], i, childs, t);
}
static void send_data_b(int i) {
  MPI_Send(perm_b + block_start[i], block_size[i], MPI_DOUBLE, who[i], 0,
           childs);
}

static void return_data_b(int p, int i) {
  MPI_Recv(perm_b + block_start[i], block_size[i], MPI_DOUBLE, p, 0, childs,
           MPI_STATUS_IGNORE);
}
static void make_who() {
  for (int i = 1; i <= num_block; i++)
    who[i] = which(i);
  vector<vector<int>> v;
  v.resize(ngpus);
  for (int i = num_block; i >= 1; i--)
    v[who[i]].push_back(i);

  int cnt = 0;
  for (int k = 0; k < (int)(v[0].size()); k++) {
    for (int p = 0; p < ngpus; p++) {
      if (int(v[p].size()) <= k)
        continue;
      send_order[cnt++] = v[p][k];
    }
  }
}
void spawn_solver_processes() {
  START("spawn_solver_processes")

  // MPI_Info info;
  // MPI_Info_create(&info);
  // MPI_Info_set(info, "hostfile", "hostfile.txt");
  // MPI_Info_set(info, "host", "a0:65,a1:33,a2:33,a3:33");
  // MPI_Info_set(info, "npernode", "32");
  // MPI_Info_set(env, "MKL_NUMTRA", "core");
  // MPI_Info_set(info, "env", "MKL_NUM_THREADS=1\nOMP_NUM_THREADS=1\n");
  // char *argv[]={"--openacc-profiling", "off", "solver_driver", nullptr};

  // MPI_Comm_spawn("solver_driver", MPI_ARGV_NULL, ngpus, info, 0,
  //                MPI_COMM_SELF, &childs, MPI_ERRCODES_IGNORE);

  MPI_Bcast(&nnz, 1, MPI_INT, MPI_ROOT, childs);
  MPI_Bcast(&cols, 1, MPI_INT, MPI_ROOT, childs);
  MPI_Bcast(&rows, 1, MPI_INT, MPI_ROOT, childs);

  MPI_Bcast(&num_block, 1, MPI_INT, MPI_ROOT, childs);

  MPI_Bcast(block_start + 1, num_block, MPI_INT, MPI_ROOT, childs);
  MPI_Bcast(block_size + 1, num_block, MPI_INT, MPI_ROOT, childs);

  make_who();
  int *mat_cnt = (int *)malloc(sizeof(int) * ngpus);
  int *nnz_pp = (int *)malloc(sizeof(int) * ngpus);
  for (int i = 0; i < ngpus; i++)
    mat_cnt[i] = 0, nnz_pp[i] = 0;

  for (int i = num_block; i >= 1; i--) {
    int p = who[i];
    mat_cnt[p]++;
    nnz_pp[p] += grid(i, i).nnz;
    for (int ii = i / 2; ii; ii /= 2) {
      mat_cnt[p] += 2;
      nnz_pp[p] += grid(i, ii).nnz;
      nnz_pp[p] += grid(ii, i).nnz;
    }
  }

  MPI_Scatter(mat_cnt, 1, MPI_INT, nullptr, 0, MPI_INT, MPI_ROOT, childs);
  MPI_Scatter(nnz_pp, 1, MPI_INT, nullptr, 0, MPI_INT, MPI_ROOT, childs);

  for (int i = num_block; i >= 1; i--) {
    int p = who[i];
    send_submatrix(p, i, i);
    for (int ii = i / 2; ii; ii /= 2) {
      send_submatrix(p, i, ii);
      send_submatrix(p, ii, i);
    }
  }

  free(nnz_pp);
  free(mat_cnt);

  END("spawn_solver_processes")
}

vector<double> solve_small(py::object A, py::array_t<double> b_py) {
  int *colidx = static_cast<int *>(
      py::cast<py::array_t<int>>(A.attr("indices")).request().ptr);
  int *rowptr = static_cast<int *>(
      py::cast<py::array_t<int>>(A.attr("indptr")).request().ptr);
  double *data = static_cast<double *>(
      py::cast<py::array_t<double>>(A.attr("data")).request().ptr);

  for (int i = 0; i < nnz; i++)
    csr_data[i] = 0;
  for (int i = 0; i < rows; i++) {
    int itr = csr_rowptr[i];
    for (int ptr = rowptr[i]; ptr < rowptr[i + 1]; ptr++) {
      int c = colidx[ptr];
      while (csr_colidx[itr] < c)
        itr++;
      csr_data[itr] = data[ptr];
    }
  }

  return _solve(csr_data, b_py);
}
vector<double> solve(py::array_t<double> Adata_py, py::array_t<double> b_py) {
  double *data = static_cast<double *>(Adata_py.request().ptr);
  return _solve(data, b_py);
}
vector<double> _solve(double *data, py::array_t<double> b_py) {
  START("solve")
  SOLVER_COMMAND command = SOLVER_RUN;
  MPI_Bcast(&command, sizeof(SOLVER_COMMAND), MPI_BYTE, MPI_ROOT, childs);

  // double *data = static_cast<doubl\e *>(Adata_py.request().ptr);
  for (int i = 0; i < nnz; i++)
    coo_val[perm_map[i]] = data[i];

  double *b = static_cast<double *>(b_py.request().ptr);
  perm_b = (double *)malloc(sizeof(double) * rows);
  for (int i = 0; i < rows; i++)
    perm_b[order[i]] = b[i];

  for (int i = num_block; i >= 1; i--) {
    send_data_b(i);
  }

  // for(int i=num_block; i>=1; i--) {
  //    send_data_merge(i);
  // }

  vector<MPI_Request> req(num_block + 1);
  for (int idx = 0; idx < num_block; idx++) {
    int i = send_order[idx];
    send_data_merge_async(i, &(req[i]));
  }
  for (int i = num_block; i >= 1; i--)
    MPI_Wait(&(req[i]), MPI_STATUS_IGNORE);

  for (int i = num_block; i >= 1; i--) {
    return_data_b(who[i], i);
  }

  vector<double> b_ret(rows);
  for (int i = 0; i < rows; i++)
    b_ret[i] = perm_b[order[i]];

  END("solve")
  return b_ret;
}

void reset_solver_processes() {
  SOLVER_COMMAND command = SOLVER_RESET;
  MPI_Bcast(&command, sizeof(SOLVER_COMMAND), MPI_BYTE, MPI_ROOT, childs);

  free(coo_r);
  free(coo_c);
  free(coo_val);
  free(perm_map);
  free(order);
  free(sizes);
  free(block_start);
  free(block_size);
  free(merge_size);
  free(send_order);
  free(perm_b);
  free(csr_rowptr);
  free(csr_colidx);
  free(csr_data);
}

void finalize_solver_processes() {
  SOLVER_COMMAND command = SOLVER_FINALIZE;
  MPI_Bcast(&command, sizeof(SOLVER_COMMAND), MPI_BYTE, MPI_ROOT, childs);

  MPI_Finalize();
  free(coo_r);
  free(coo_c);
  free(coo_val);
  free(perm_map);
  free(order);
  free(sizes);
  free(block_start);
  free(block_size);
  free(merge_size);
  free(send_order);
  free(perm_b);
  free(csr_rowptr);
  free(csr_colidx);
  free(csr_data);
}

////////////////////////////////////////////////////////////
/////                                                   ////
/////             Local Static Functions                ////
/////                                                   ////
////////////////////////////////////////////////////////////

static void spawn_parmetis_processes(int parts, int nnz, int rows) {
  START("spawn_parmetis_processes")

  MPI_Bcast(&nnz, 1, MPI_INT, MPI_ROOT, childs);
  MPI_Bcast(&rows, 1, MPI_INT, MPI_ROOT, childs);

  vtxdist = (int *)malloc(sizeof(int) * (parts + 1));
  loc_rowss = (int *)malloc(sizeof(int) * (parts));
  displs = (int *)malloc(sizeof(int) * (parts));
  loc_nnzs = (int *)malloc(sizeof(int) * (parts));

  END("spawn_parmetis_processes")
}

static void finalize_parmetis_processes() {
  PARMETIS_COMMAND command = PARMETIS_FINALIZE;
  MPI_Bcast(&command, sizeof(PARMETIS_COMMAND), MPI_BYTE, MPI_ROOT, childs);

  free(loc_nnzs);
  free(displs);
  free(loc_rowss);
  free(vtxdist);
}

static csc_matrix addTransposeEliminateDiag(csc_matrix A, int *buf1,
                                            int *buf2) {
  // assert rows=cols;
  int rows = A.n, cols = A.m, nnz = A.nnz, *rowidx = A.rowidx,
      *colptr = A.colptr;
  vector<int> rowidx_t(nnz);
  vector<int> colptr_t(cols + 1, 0);

  for (int i = 0; i < nnz; i++)
    colptr_t[rowidx[i]]++;
  for (int i = 0; i < cols; i++)
    colptr_t[i + 1] += colptr_t[i];

  for (int c = cols - 1; c >= 0; c--) {
    for (int i = colptr[c]; i < colptr[c + 1]; i++) {
      int r = rowidx[i];
      colptr_t[r]--;
      rowidx_t[colptr_t[r]] = c;
    }
  }

  int new_nnz = 0;
  int *new_colptr = buf1;
  int *new_rowidx = buf2;

  for (int c = 0; c < cols; c++) {
    new_colptr[c] = new_nnz;
    int l1 = colptr[c], r1 = colptr[c + 1], l2 = colptr_t[c],
        r2 = colptr_t[c + 1];
    while (l1 < r1 && l2 < r2) {
      if (rowidx[l1] < rowidx_t[l2]) {
        if (rowidx[l1] != c)
          new_rowidx[new_nnz++] = rowidx[l1];
        l1++;
      } else if (rowidx[l1] > rowidx_t[l2]) {
        if (rowidx_t[l2] != c)
          new_rowidx[new_nnz++] = rowidx_t[l2];
        l2++;
      } else {
        if (rowidx[l1] != c)
          new_rowidx[new_nnz++] = rowidx[l1];
        l1++, l2++;
      }
    }
    while (l1 < r1) {
      if (rowidx[l1] != c)
        new_rowidx[new_nnz++] = rowidx[l1];
      l1++;
    }
    while (l2 < r2) {
      if (rowidx_t[l2] != c)
        new_rowidx[new_nnz++] = rowidx_t[l2];
      l2++;
    }
  }
  new_colptr[cols] = new_nnz;

  return {rows, cols, new_nnz, new_rowidx, new_colptr, nullptr};
}

// Call Parmetis for csr_matrix A. Since A must be symmetric, csc_matrix is also
// allowed.
static void call_parmetis(csc_matrix A, int parts, int *order, int *sizes) {
  PARMETIS_COMMAND command = PARMETIS_RUN;
  MPI_Bcast(&command, sizeof(PARMETIS_COMMAND), MPI_BYTE, MPI_ROOT, childs);
  int rows = A.n, cols = A.m, nnz = A.nnz, *colidx = A.rowidx,
      *rowptr = A.colptr;

  MPI_Bcast(&rows, 1, MPI_INT, MPI_ROOT, childs);
  MPI_Bcast(&cols, 1, MPI_INT, MPI_ROOT, childs);
  MPI_Bcast(&nnz, 1, MPI_INT, MPI_ROOT, childs);

  for (int i = 0; i <= parts; i++)
    vtxdist[i] = (int)(1ll * rows * i / parts);

  for (int i = 0; i < parts; i++)
    loc_rowss[i] = vtxdist[i + 1] - vtxdist[i] + 1;

  MPI_Scatterv(rowptr, loc_rowss, vtxdist, MPI_INT, nullptr, 0, MPI_INT,
               MPI_ROOT, childs);

  for (int i = 0; i < parts; i++)
    displs[i] = *(rowptr + vtxdist[i]);
  for (int i = 0; i < parts; i++)
    loc_nnzs[i] = *(rowptr + vtxdist[i + 1]) - *(rowptr + vtxdist[i]);
  MPI_Scatterv(colidx, loc_nnzs, displs, MPI_INT, nullptr, 0, MPI_INT, MPI_ROOT,
               childs);

  // Change to MPIGatherv?
  for (int i = 0; i < parts; i++) {
    MPI_Recv(order + vtxdist[i], loc_rowss[i], MPI_INT, i, 0, childs,
             MPI_STATUS_IGNORE);
  }

  MPI_Recv(sizes, parts + parts - 1, MPI_INT, 0, 0, childs, MPI_STATUS_IGNORE);
}

static csc_matrix permutate(csc_matrix A, int *order, int *buf1,
                            int *buf2) { //& construct
  int *indices = A.rowidx;
  int *indptr = A.colptr;
  int rows = A.n, cols = A.m, nnz = A.nnz;

  int sum = 0;

  int *new_indptr = buf1;
  int *new_indices = buf2;

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
  return {rows, cols, nnz, new_indices, new_indptr, nullptr};
}
static void fill(int *sizes, int *subsizes, int head, int parts) {
  int cnt = 0;
  for (int i = parts; i >= 1; i /= 2) {
    for (int j = i - 1; j >= 0; j--) {
      sizes[(parts * parts * 2 - 1) - (head * i + j)] = subsizes[cnt++];
    }
  }
}

static void dfs_block_order(vector<int> &v, int cur, int parts) {
  if (parts > 1) {
    dfs_block_order(v, cur * 2 + 1, parts / 2);
    dfs_block_order(v, cur * 2, parts / 2);
  }
  v.push_back(cur);
}
static void calculate_block(int *sizes, int parts) {
  vector<int> block_order;
  dfs_block_order(block_order, 1, parts);
  num_block = parts + parts - 1;

  int cur = 0;
  for (auto &e : block_order) {
    block_start[e] = cur;
    block_size[e] = sizes[parts + parts - 1 - e];
    cur += block_size[e];
  }
}
static csc_matrix getSubmatrix(csc_matrix A, int i, int j, int parts, int *buf1,
                               int *buf2) {
  int *indices = A.rowidx;
  int *indptr = A.colptr;
  // int rows=A.n, cols=A.m, nnz=A.nnz;

  int *new_colptr = buf1;
  int new_nnz = 0;

  new_colptr[0] = 0;
  for (int c = block_start[j]; c < block_start[j] + block_size[j]; c++) {
    for (int ri = indptr[c]; ri < indptr[c + 1]; ri++) {
      int r = indices[ri];
      if (r < block_start[i])
        continue;
      if (r >= block_start[i] + block_size[i])
        break;
      new_nnz++;
    }
    new_colptr[c + 1 - block_start[j]] = new_nnz;
  }
  int *new_rowidx = buf2;

  int idx = 0;
  for (int c = block_start[j]; c < block_start[j] + block_size[j]; c++) {
    for (int ri = indptr[c]; ri < indptr[c + 1]; ri++) {
      int r = indices[ri];
      if (r < block_start[i])
        continue;
      if (r >= block_start[i] + block_size[i])
        break;
      new_rowidx[idx++] = r - block_start[i];
    }
  }
  return {block_size[i], block_size[j], new_nnz, new_rowidx, new_colptr};
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

// TODO currently, A format is csc but contain data for csr
static void construct(csc_matrix A) {
  int *indices = A.rowidx;
  int *indptr = A.colptr;
  int cols = A.m, nnz = A.nnz;
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
