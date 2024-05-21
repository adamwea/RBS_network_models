#!/bin/bash

# Compile ring_shmem.c
mpicc ring_oshmem_c.c -o ring_oshmem -L/opt/openmpi/lib -loshmem

# Path to the compiled OpenSHMEM program
SHMEM_PROGRAM="./ring_oshmem"

# Function to run the test
run_test() {
  # Number of processes to use
  PROCESSES=$1

  # Print debug information
  echo "Running oshrun -np $PROCESSES $SHMEM_PROGRAM"

  # Run the OpenSHMEM program using oshrun
  OUTPUT=$(oshrun -np $PROCESSES $SHMEM_PROGRAM 2>&1)

  # Print the output
  echo "Output of oshrun -np $PROCESSES $SHMEM_PROGRAM:"
  echo "$OUTPUT"

  # Verify that the expected output is present
  EXPECTED_OUTPUT1="Process 0 puts message 10 to 1"
  EXPECTED_OUTPUT2="Process 1 exiting"
  EXPECTED_OUTPUT3="Process 0 exiting"

  if echo "$OUTPUT" | grep -q "$EXPECTED_OUTPUT1" && echo "$OUTPUT" | grep -q "$EXPECTED_OUTPUT2" && echo "$OUTPUT" | grep -q "$EXPECTED_OUTPUT3"; then
    echo "Test passed: Expected output found."
  else
    echo "Test failed: Expected output not found."
  fi
}

# Example usage
# Run ring_shmem on 2 processes (minimal for this test)
run_test 2


