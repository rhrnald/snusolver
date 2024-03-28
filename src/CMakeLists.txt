find_package(pybind11 REQUIRED)

set(MPI_ROOT $ENV{MPI_ROOT})
#set(CMAKE_C_COMPILER "${MPI_ROOT}/bin/mpicc")
#set(CMAKE_CXX_COMPILER "${MPI_ROOT}/bin/mpicxx")
set(CMAKE_C_COMPILER "mpicc")
set(CMAKE_CXX_COMPILER "mpicxx")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_FLAGS "-O3 -std=c++14 -g -Wall -Wno-unused-result -Wl,--no-as-needed -fopenmp -m64")
set(CMAKE_CUDA_FLAGS "-O3 -std=c++14 -g -Wall -Wno-unused-result -Wl,--no-as-needed -fopenmp -m64")

# Setup CUDA
if(NOT DEFINED ENV{CUDA_ROOT})
    message(FATAL_ERROR "CUDA_ROOT environment variable is REQUIRED")
endif()
set(CUDA_ROOT $ENV{CUDA_ROOT})
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
include_directories(${CUDA_ROOT}/include)
link_directories(${CUDA_ROOT}/lib64)
set(CMAKE_CUDA_ARCHITECTURES 70 80)
# find_package(CUDA)

# Setup SuperLU
# set(SUPERLU_ROOT $ENV{SUPERLU_ROOT})
# include_directories(${SUPERLU_ROOT}/SRC)
# link_directories(${SUPERLU_ROOT}/build/SRC)

# Setup MKLBLAS
set(MKL_ROOT $ENV{MKL_ROOT})
include_directories(${MKL_ROOT}/include)
link_directories(${MKL_ROOT}/lib/intel64)
#add_compile_options(-DMKL_ILP64 -m64)
add_compile_options(-m64)
add_link_options(-Wl,--no-as-needed)

# Setup OpenMP
set(IOMP_ROOT $ENV{IOMP_ROOT})
# include_directories(${IOMP_ROOT}/include)
link_directories(${IOMP_ROOT}/lib/intel64_lin)
link_directories(${IOMP_ROOT}/lib/)

# Setup MPI
set(MPI_ROOT $ENV{MPI_ROOT})
include_directories(${MPI_ROOT}/include)
link_directories(${MPI_ROOT}/lib)

# Setup parmetis
set(PARMETIS_ROOT $ENV{PARMETIS_ROOT})
include_directories(${PARMETIS_ROOT}/include)
include_directories(${PARMETIS_ROOT}/metis/include)
link_directories(${PARMETIS_ROOT}/build/Linux-x86_64/libparmetis)
link_directories(${PARMETIS_ROOT}/build/Linux-x86_64/libmetis)

#pybind11_add_module(snusolver_c snusolver_c.cpp)
#target_link_libraries(snusolver_c PUBLIC mpi mpi_cxx)

add_executable(driver driver.cpp parmetis_driver.cpp solver.cu kernel.cpp kernel_gpu.cu read_matrix.cpp mmio.cpp)
target_link_libraries(driver parmetis metis)
target_link_libraries(driver mpi mpi_cxx)
target_link_libraries(driver iomp5 pthread m dl cublas cudart cuda curand cusolver nvToolsExt)
target_link_libraries(driver mkl_scalapack_lp64 mkl_intel_lp64 mkl_intel_thread mkl_core mkl_blacs_intelmpi_lp64 mkl_gf_lp64 mkl_gnu_thread)

# add_executable(parmetis_driver parmetis_driver.cpp solver_driver.cu kernel.cpp kernel_gpu.cu)
# target_link_libraries(parmetis_driver parmetis metis)
# target_link_libraries(parmetis_driver mpi mpi_cxx)
# target_link_libraries(parmetis_driver iomp5 pthread m dl cublas cudart cuda curand cusolver nvToolsExt)
# target_link_libraries(parmetis_driver mkl_scalapack_lp64 mkl_intel_lp64 mkl_intel_thread mkl_core mkl_blacs_intelmpi_lp64 mkl_gf_lp64 mkl_gnu_thread)

# add_executable(solver_driver solver_driver.cpp kernel.cpp kernel_sparse.cpp)
# target_link_libraries(solver_driver mpi mpi_cxx)
# target_link_libraries(solver_driver iomp5 pthread m dl)
# target_link_libraries(solver_driver superlu_dist parmetis metis)
# target_link_libraries(solver_driver mkl_scalapack_lp64 mkl_intel_lp64 mkl_intel_thread mkl_core mkl_blacs_intelmpi_lp64 mkl_gf_lp64 mkl_gnu_thread mkl_def mkl_avx)

# add_executable(solver_driver_gpu solver_driver_gpu.cu kernel_gpu.cu)
# target_link_libraries(solver_driver_gpu mpi mpi_cxx)
# target_link_libraries(solver_driver_gpu iomp5 pthread m dl cublas cudart cuda curand cusolver nvToolsExt)


# Make sure to match the MKL link options with SuperLU_DIST build options
# -DTPL_BLAS_LIBRARIES="${MKL_ROOT}/lib/intel64/libmkl_gf_lp64.so;${MKL_ROOT}/lib/intel64/libmkl_gnu_thread.so;${MKL_ROOT}/lib/intel64/libmkl_core.so;${MKL_ROOT}/lib/intel64/libmkl_def.so;${MKL_ROOT}/lib/intel64/libmkl_avx.so" \

#mpicxx test.cpp -L${MKLROOT}/lib/intel64 -L${IOMP_ROOT}/lib/intel64_lin -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -liomp5 -lpthread -lm -ldl -I${MKLROOT}/include 
