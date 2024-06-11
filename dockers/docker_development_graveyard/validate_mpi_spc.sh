#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Source code file name
SOURCE_FILE="spc_example.c"

# Output executable name
EXECUTABLE="spc_example"

# Compile the MPI program
mpicc $SOURCE_FILE -o $EXECUTABLE

# Function to run the test
run_test() {
  # Number of messages to send
  NUM_MESSAGES=$1
  
  # Size of each message
  MESSAGE_SIZE=$2
  
  # Print debug information
  echo "Running mpirun -n 2 --mca mpi_spc_attach all --mca mpi_spc_dump_enabled true ./$EXECUTABLE $NUM_MESSAGES $MESSAGE_SIZE"
  
  # Run the MPI program using mpirun
  mpirun -n 2 --mca mpi_spc_attach all --mca mpi_spc_dump_enabled true ./$EXECUTABLE $NUM_MESSAGES $MESSAGE_SIZE 2>&1
  OUTPUT=$(mpirun -n 2 --mca mpi_spc_attach all --mca mpi_spc_dump_enabled true ./$EXECUTABLE $NUM_MESSAGES $MESSAGE_SIZE 2>&1)
  
  # Print the output
  echo "Output of mpirun -n 2 ./$EXECUTABLE $NUM_MESSAGES $MESSAGE_SIZE:"
  echo "$OUTPUT"
  
  # Verify that the expected output is present
  if echo "$OUTPUT" | grep -q "Value Read:"; then
    echo "Test passed: Expected output found."
  else
    echo "Test failed: Expected output not found."
  fi
}

# Example usage
# Run spc_example with 10 messages of size 1024 bytes
run_test 10 1024
