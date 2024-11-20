'''Main Python Script for Running Network Simulations'''

'''Setup Python environment for running the script'''
from pprint import pprint
import setup_environment
setup_environment.set_pythonpath()

'''
Import Local Modules

Note: Edit the settings in parse_user_args.py to adjust runtime settings of evolutionary algorithm.
'''
from simulate._config_files import parse_kwargs
from simulate._config_files import evolutionary_parameter_space
from simulate._temp_files.temp_user_args import *