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

# 1 Node
# for MAT in $REFERENCE_MATS
# do  
#     echo "Running $MAT using 1 node"
#     salloc --nodes 1 \
#         --partition=PA \
#         --exclusive \
#         --job-name=solver \
#         --nodelist=a1 \
#         mpirun -np 32 \
#         -x PATH \
#         -host a1:32 \
#         -x UCX_TLS=rc,cuda \
#         -mca pml ucx -mca btl ^openib \
#         ./driver $SUITESPARSE_MATRIX_DIR/$MAT/$MAT.mtx
#     echo "Done"
#     echo $'\n'
# done

# 4 Node
# for MAT in $TEST_MATS
# do  
#     echo "Running $MAT using 4 nodes"
#     salloc --nodes 4 \
#         --partition=PA \
#         --exclusive \
#         --job-name=solver \
#         --nodelist=a0,a1,a2,a3 \
#         mpirun -np 128 \
#         -x PATH \
#         -host a0:32,a1:32,a2:32,a3:32 \
#         -mca pml ucx -mca btl ^openib \
#         ./driver $SUITESPARSE_MATRIX_DIR/$MAT/$MAT.mtx
#     echo "Done"
#     echo $'\n'
# done


REFERENCE_MATS="
G3_circuit
"

# Reference Performance
for NODE in a0 a1 a2 a3 c0 c1 c2 c3 c5 c6
do
    for MAT in $REFERENCE_MATS
    do  
        echo "Running $MAT using $NODE node"
        salloc --nodes 1 \
            --partition=CPU8 \
            --exclusive \
            --job-name=solver \
            --nodelist=$NODE \
            mpirun -np 32 \
            -x PATH \
            -host $NODE:32 \
            -mca pml ucx -mca btl ^openib \
            ./driver $SUITESPARSE_MATRIX_DIR/$MAT/$MAT.mtx
        echo "Done"
        echo $'\n'
    done
done