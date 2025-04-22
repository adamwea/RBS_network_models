import os
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.run_sorter import run_sorter
import glob

# =============================================================================
script_path = os.path.abspath(__file__)
#os.chdir(os.path.dirname(script_path))
raw_data_path = (
    os.path.abspath( 
    #this should be an .h5 file
    #'/pscratch/sd/a/adammwea/workspace/_raw_data/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/data.raw.h5',
    #'/pscratch/sd/a/adammwea/workspace/_raw_data/Organoid_RTT_R270X_pA_pD_B1_d91/250107/M07297/Network/000028/data.raw.h5',
    
    # aw 2025-03-06 14:26:03 - Roy asked me to generate bursting plots for these data across DIVs - going to do these remaining DIVs one at a time.
    
    
    ))

#sorted_output_dir = '**/data/CDKL5/DIV21/sorted'   #syntax for glob.glob
sorted_output_dir = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/sorted'
sorted_output_dir = glob.glob(sorted_output_dir, recursive=True)[0]
waveform_output_dir = os.path.join(os.path.dirname(sorted_output_dir), 'waveforms')
 
# =============================================================================
run_sorter(
    raw_data_path,
    sorted_output_dir,
    waveform_output_dir,
    use_docker=False,   # NOTE: Default is True. Comment out this line to use docker.
                        #       If running on NERSC, you'll need to run without docker and with shifter.
                        #       see below for shifter command to run on NERSC
    #try_load = False,   # NOTE: Default is True. Comment out this line to try loading the sorted data.
    )

# =============================================================================
# bash command to run kilosort2 on NERSC
# TODO: Update shifter to include installed editable packages. RBS_network_models and MEA_Analysis.

#  to run spikesorting as needed in interactive node with gpu:
'''
salloc -A m2043_g -q interactive -C gpu -t 04:00:00 --nodes=1 --gpus=1 --image=adammwea/axonkilo_docker:v7
'''
# after salloc, run the following command: # NOTE: replace path to script as needed.
'''
shifter --image=adammwea/axonkilo_docker:v7 /bin/bash
pip install -e /pscratch/sd/a/adammwea/workspace/RBS_network_models
pip install -e /pscratch/sd/a/adammwea/workspace/MEA_Analysis
# python /pscratch/sd/a/adammwea/workspace/RBS_network_models/scripts/CDKL5/spikesort_DIV21_WT.py \
#     > /pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/sorted/logs/test_spikesort.log 2>&1

# create dirs for sorted data as needed
mkdir -p /pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/sorted/logs
python /pscratch/sd/a/adammwea/workspace/RBS_network_models/scripts/Organoid_RTT_R270X/DIV112_WT/spikesort.py \
    > /pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/sorted/logs/test_spikesort.log 2>&1
'''