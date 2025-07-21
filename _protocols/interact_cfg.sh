# bin/bash

# updated # aw 2025-04-21 03:36:06
module load conda
conda activate my_mpi4py_env

# uncomment as needed - just need to reinstall since i moved repos
# pip install -e /global/homes/a/adammwea/dev/RBS_network_models
# pip install -e /global/homes/a/adammwea/dev/netpyne
# pip install -e /global/homes/a/adammwea/dev/MEA_Analysis

# load the correct compiler environment
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
echo "If you see 'numprocs=1' above, then mpi cfg is successful."

python /global/homes/a/adammwea/dev/RBS_network_models/_scripts/run_permutations.py