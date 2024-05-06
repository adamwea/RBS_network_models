#bash NERSC/neuron_run.sh
module load conda
conda activate 2DSims

# Set the library path for NEURON and Python libraries
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/neuron/lib:$LD_LIBRARY_PATH

# Set the path to include the NEURON binaries
export PATH=$HOME/neuron/bin:$PATH

#check the versions of the installed software
nrniv --version
echo "Plotting..."

python3 NERSC/plot_simData.py
#python3 NERSC/generate_pdf_report.py



    