#!bin/bash

SUITESPARSE_MATRIX_DIR="$HOME/project/samsung-display/mats/suitesparse-matrix/mtx"

TEST_MATS="
lung2
twotone
xenon2
c-73
scircuit
ohne2
ss1
HTC_336_4438
Raj1
ASIC_320ks
ASIC_680ks
tmt_unsym
ecology1
webbase-1M
thermal2
G3_circuit
memchip
Freescale1
circuit5M_dc
rajat31
"

PARTITION=PB
NODE=b2

for MAT in $TEST_MATS
do  
    echo "Running $MAT using 1 node"
    salloc --nodes 1 \
        --partition=$PARTITION \
        --exclusive \
        --job-name=solver \
        --nodelist=$NODE \
        mpirun -np 32 \
        -x PATH \
        -host $NODE:32 \
        -x UCX_TLS=rc,cuda \
        -mca pml ucx -mca btl ^openib \
        ./driver $SUITESPARSE_MATRIX_DIR/$MAT/$MAT.mtx
    echo "Done"
    echo $'\n'
done
