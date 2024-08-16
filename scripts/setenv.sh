# Need to modify considering your environment
ROOTDIR=${HOME}/project/samsung-display/pardiso-thunder
export CUDA_ROOT=/usr/local/cuda
export MPI_ROOT=${HOME}/lib/openmpi
export MKL_ROOT=${HOME}/intel/oneapi/mkl/latest
export IOMP_ROOT=${HOME}/intel/oneapi/compiler/latest/linux/compiler

# Do not need to modify
export PARMETIS_ROOT=$ROOTDIR/../parmetis-4.0.3
export LD_LIBRARY_PATH=$MPI_ROOT/lib:$LD_LIBRARY_PATH
export PATH=$MPI_ROOT/bin:$PATH
export PYTHONPATH=$ROOTDIR/snusolver:$PYTHONPATH
export PYTHONPATH=$ROOTDIR/build:$PYTHONPATH
export PATH=$ROOTDIR/build:$PATH
