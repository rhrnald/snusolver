#include "parmetis_driver.h"
#include "solver_driver.h"

int main(int argc, char *argv[]) {
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &omp_mpi_level);
  while (true) {
    main_parmetis();
    int ret = main_solver();
    if (!ret)
      break;
  }
}
