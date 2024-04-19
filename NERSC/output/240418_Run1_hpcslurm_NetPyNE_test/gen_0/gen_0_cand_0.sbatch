#!/bin/bash
#SBATCH --job-name=gen_0_cand_0
#SBATCH -A default
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH -o /global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240418_Run1_hpcslurm_NetPyNE_test/gen_0/gen_0_cand_0.run
#SBATCH -e /global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240418_Run1_hpcslurm_NetPyNE_test/gen_0/gen_0_cand_0.err
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=end

#SBATCH -q debug

module load python
module load conda
module load openmpi
conda activate 2DSims

source ~/.bashrc
cd .
mpirun -n 256 nrniv -python -mpi /global/u2/a/adammwea/2DNetworkSimulations/NERSC/init.py simConfig=/global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240418_Run1_hpcslurm_NetPyNE_test/gen_0/gen_0_cand_0_cfg.json netParams=/global/u2/a/adammwea/2DNetworkSimulations/NERSC/output/240418_Run1_hpcslurm_NetPyNE_test/240418_Run1_hpcslurm_NetPyNE_test_netParams.py 
wait
    