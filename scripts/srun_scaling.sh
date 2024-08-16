#!bin/bash

SUITESPARSE_MATRIX_DIR="$HOME/project/samsung-display/mats/suitesparse-matrix/mtx"

TEST_MATS="
twotone
Raj1
ASIC_320ks
ASIC_680ks
tmt_unsym
ecology1
G3_circuit
memchip
Freescale1
circuit5M_dc
rajat31
"

TEST_MATS2="
ecology1
G3_circuit
"

PARTITION=PB


for MAT in $TEST_MATS2
do  
    # 1 Node (1 x 1, 1 x 2, 1 x 4, 1 x 8, 1 x 16, 1 x 32)
    for i in 16 32
    do
        echo "Running $MAT using 1 x $i Processes"
        salloc --nodes 1 \
            --partition=$PARTITION \
            --exclusive \
            --job-name=solver \
            --nodelist=b2 \
            mpirun -np $i \
            -x PATH \
            -host b2:$i \
            -x UCX_TLS=rc,cuda \
            -mca pml ucx -mca btl ^openib \
            ./driver $SUITESPARSE_MATRIX_DIR/$MAT/$MAT.mtx
        echo "Done"
        echo $'\n'
    done

    # 2 Node (2 x 32)
    echo "Running $MAT using 2 x 32 Processes"
    salloc --nodes 2 \
        --partition=$PARTITION \
        --exclusive \
        --job-name=solver \
        --nodelist=b2,b3 \
        mpirun -np 64 \
        -x PATH \
        -host b2:32,b3:32 \
        -mca pml ucx -mca btl ^openib \
        ./driver $SUITESPARSE_MATRIX_DIR/$MAT/$MAT.mtx
    echo "Done"
    echo $'\n'

    # 4 Node (4 x 32)
    # echo "Running $MAT using 4 x 32 Processes"
    # salloc --nodes 4 \
    #     --partition=$PARTITION \
    #     --exclusive \
    #     --job-name=solver \
    #     --nodelist=b0,b2,b3,b5 \
    #     mpirun -np 128 \
    #     -x PATH \
    #     -host b0:32,b2:32,b3:32,b5:32 \
    #     -mca pml ucx -mca btl ^openib \
    #     ./driver $SUITESPARSE_MATRIX_DIR/$MAT/$MAT.mtx
    # echo "Done"
    # echo $'\n'
done
