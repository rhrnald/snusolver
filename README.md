# SnuSolver

## Setup & Build
```bash
bash setenv.sh
cmake .. && make -j
```

## Run
```bash
# Run on four-node cluster (32 Processes per node)
mpirun -mca pml ucx -mca btl ^openib -x UCX_TLS=rc,cuda \
    -np 128 -H <host1>:32,<host2>:32,<host3>:32,<host4>:32 \
    ./driver </path/to/matrix/mtx>
```