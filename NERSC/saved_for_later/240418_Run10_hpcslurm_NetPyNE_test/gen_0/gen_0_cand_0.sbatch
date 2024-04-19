#!/bin/bash
#SBATCH --job-name=gen_0_cand_0
#SBATCH -A m2043
#SBATCH -t 00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH -o /global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240418_Run10_hpcslurm_NetPyNE_test/gen_0/gen_0_cand_0.run
#SBATCH -e /global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240418_Run10_hpcslurm_NetPyNE_test/gen_0/gen_0_cand_0.err
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=end


##Custom SLURM Options
#SBATCH -q debug
#SBATCH -C cpu

module load python
module load conda

#
# cray-mpich and cray-libsci conflict with openmpi so unload them
#
module unload cray-mpich
module unload cray-libsci
module use /global/common/software/m3169/perlmutter/modulefiles
module load openmpi

conda activate 2DSims

touch ~/.bashrc

# export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread
##Custom SLURM Options

source ~/.bashrc
cd .
mpirun -n 256 nrniv -python -mpi /global/u2/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240418_Run10_hpcslurm_NetPyNE_test/gen_0/gen_0_cand_0_cfg.json netParams=/global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240418_Run10_hpcslurm_NetPyNE_test/240418_Run10_hpcslurm_NetPyNE_test_netParams.py 
wait
    