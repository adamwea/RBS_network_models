#!/bin/bash

# Test case: Use mpirun to launch a trivial MPI program (hello) that does no MPI communication.

#compile hello.c
mpicc -o hello hello_c.c

# Path to the compiled MPI program
MPI_PROGRAM="./hello"

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

  # Verify that the program was executed by the specified number of processes
  PROCESS_COUNT=$(echo "$OUTPUT" | grep -c "Hello, world, I am")

  if [ "$PROCESS_COUNT" -eq "$PROCESSES" ]; then
    echo "Test passed: Program executed on $PROCESS_COUNT processes."
  else
    echo "Test failed: Expected $PROCESSES processes, but got $PROCESS_COUNT processes."
  fi
}

# Example usage
# Run hello on 4 processes
run_test 4

# Run hello on 8 processes
run_test 8