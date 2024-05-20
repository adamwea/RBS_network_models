#!/bin/bash

# Test case: Use oshrun to launch a trivial MPI program (hello_shmem) that does no OpenSHMEM communication.

# Compile hello_shmem.c
mpicc -o hello_oshmem hello_oshmem_c.c

# Path to the compiled MPI program
MPI_PROGRAM="./hello_oshmem"

# Function to run the test
run_test() {
  # Number of processes to use
  PROCESSES=$1

  # Print debug information
  echo "Running oshrun -np $PROCESSES $MPI_PROGRAM"

  # Run the MPI program using oshrun
  OUTPUT=$(oshrun -np $PROCESSES $MPI_PROGRAM 2>&1)

  # Print the output
  echo "Output of oshrun -np $PROCESSES $MPI_PROGRAM:"
  echo "$OUTPUT"

  # Verify that the program was executed by the specified number of processes
  PROCESS_COUNT=$(echo "$OUTPUT" | grep -c "Hello, world, I am")

  if [ "$PROCESS_COUNT" -eq "$PROCESSES" ]; then
    echo "Test passed: Program executed on $PROCESS_COUNT processes."
  else
    echo "Test failed: Expected $PROCESSES processes, but got $PROCESS_COUNT processes."
  fi
}

# Example usage
# Run hello_shmem on 4 processes
run_test 4

# Run hello_shmem on 8 processes
run_test 8
