# this test script is for testing typical use case of extract_features.py
import os
from RBS_network_models import extract_features as ef
from RBS_network_models.CDKL5.DIV21.src import conv_params # NOTE: this works...on login node in NERSC.
                                                            #      But I think I remember that it didn't work on local machine. laptop.
                                                            #       TODO: test on local machine.
# =============================================================================
script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(script_path))
raw_data_paths = [ 
    # NOTE: this is a list of paths to raw data files that you want to extract features from
    #      this is useful for batch processing.
    #      Also, NOTE: if parent dirs are provided, each path will be searched recursively for .h5 files
    #'../data/raw/CDKL5-E6D_T2_C1_05212024/240611/M08034/Network/000096/data.raw.h5',
    '../data/raw/CDKL5-E6D_T2_C1_05212024/240611/M06844/Network/000076/data.raw.h5'
]
sorted_data_dir = (
    # this should be the parent directory of all sorted data files, ideally following proper data structure, naming conventions
    '../data/CDKL5/DIV21/sorted'   
)
# =============================================================================
'''main'''
#conv_params = CDKL5.DIV21.src.conv_params
network_metrics_output_dir = os.path.join(os.path.dirname(sorted_data_dir), 'network_metrics')
feature_data = ef.extract_network_features(
    raw_data_paths,
    sorted_data_dir=sorted_data_dir,
    output_dir = network_metrics_output_dir,
    stream_select=None,
    conv_params=conv_params,
    # plot=True,
    )
print("Network metrics saved.")