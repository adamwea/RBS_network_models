#general imports
import os
import platform
import re
import logging
from pprint import pprint

#spikeinterface imports
import spikeinterface
import spikeinterface.sorters as ss
import spikeinterface.full as si

#local imports
from file_selector import main as file_selector_main
#from AxonReconPipeline.src.archive.extract_raw_assay_data import main as extract_raw_assay_data_main
from generate_recording_and_sorting_objects import main as generate_recording_and_sorting_objects
from axon_trace_classes import axon_trace_objects
from reconstruct_axons import reconstruct_axon_trace_objects
import mea_processing_library as MPL
import helper_functions as helper
import spike_sorting as hp #hornauerp, https://github.com/hornauerp/axon_tracking/blob/main/axon_tracking/spike_sorting.py
import template_extraction as te_hp

#Logger Setup
#Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
stream_handler = logging.StreamHandler()  # logs to console
#file_handler = logging.FileHandler('file.log')  # logs to a file

# Set level of handlers
stream_handler.setLevel(logging.DEBUG)
#file_handler.setLevel(logging.ERROR)

# Add handlers to the logger
logger.addHandler(stream_handler)
#logger.addHandler(file_handler)

# Create formatters and add it to handlers
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(module)s.%(funcName)s')
stream_handler.setFormatter(formatter)