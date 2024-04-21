# user_inputs.py
import sys

from USER_INPUTS import *

print(f'USER_INPUTs:')
def print_globals():
    for name, value in globals().items():
        # Ignore built-in and imported variables
        if not name.startswith("__") and not isinstance(value, type(sys)):
            print(f"{name}={value}")

# Call the function
print_globals()