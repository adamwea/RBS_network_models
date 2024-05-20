#!/bin/bash

# Test case: Use mpirun to launch a trivial MPI program (ring) that sends and receives a few MPI messages.

#compile ring.c
mpicc -o ring ring_c.c

# Path to the compiled MPI program
MPI_PROGRAM="./ring"

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

  # Verify that each process exited
  EXPECTED_MESSAGES=$PROCESSES
  EXIT_MESSAGES=$(echo "$OUTPUT" | grep -c "exiting")

  if [ "$EXIT_MESSAGES" -eq "$EXPECTED_MESSAGES" ]; then
    echo "Test passed: Program executed with $EXIT_MESSAGES processes exiting."
  else
    echo "Test failed: Expected $EXPECTED_MESSAGES processes to exit, but got $EXIT_MESSAGES exits."
  fi
}

# Example usage
# Run ring on 4 processes
run_test 4

# Run ring on 8 processes
run_test 8
