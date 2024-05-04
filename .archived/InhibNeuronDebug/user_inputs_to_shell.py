# user_inputs.py
import sys

from USER_INPUTS import *

print(f'USER_INPUTs:')
def print_globals():
    for name, value in globals().items():
        # Ignore built-in and imported variables and only print variables starting with "USER"
        if not name.startswith("__") and not isinstance(value, type(sys)) and name.startswith("USER"):
            print(f"{name}={value}")

# Call the function
print_globals()