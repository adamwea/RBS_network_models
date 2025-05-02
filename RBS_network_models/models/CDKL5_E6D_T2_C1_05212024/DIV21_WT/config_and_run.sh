# bin/bash

# updated # aw 2025-04-21 03:36:06
module load conda
conda activate my_mpi4py_env
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load cray-mpich
echo $MPICH_DIR

# two main subdirs under $MPICH_DIR hold the .so files:
export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH

# tell NEURON exactly which libmpi to dlopen:
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)

# validate setup
nrniv -mpi -python - <<EOF
from neuron import h
print("MPI load OK, h =", h)
EOF

#
python /global/homes/a/adammwea/workspace/repos/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_WT/run_batch.py