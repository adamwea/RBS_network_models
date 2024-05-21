#!/bin/bash

# Compile connectivity_test.c
mpicc connectivity_c.c -o connectivity

# Path to the compiled MPI program
MPI_PROGRAM="./connectivity"

# Function to run the test
run_test() {
  # Number of processes to use
  PROCESSES=$1

  # Print debug information
  echo "Running mpirun -np $PROCESSES $MPI_PROGRAM"

  # Run the MPI program using mpirun
  OUTPUT=$(mpirun -np $PROCESSES $MPI_PROGRAM 2>&1)

  # Print the output
  echo "Output of mpirun -np $PROCESSES $MPI_PROGRAM:"
  echo "$OUTPUT"

  # Verify that the expected output is present
  EXPECTED_OUTPUT="Connectivity test on $PROCESSES processes PASSED."
  if echo "$OUTPUT" | grep -q "$EXPECTED_OUTPUT"; then
    echo "Test passed: Expected output found."
  else
    echo "Test failed: Expected output not found."
  fi
}

# Example usage
# Run connectivity_test on 4 processes
run_test 4
