import logging
import os

def setup_logger():
    '''Sets up logging for the script.'''
    script_dir = os.path.dirname(os.path.realpath(__file__))
    log_file = f'{script_dir}/batchRun.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    return logger