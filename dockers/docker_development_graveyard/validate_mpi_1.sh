#!/bin/bash

# Test case: Use mpirun to launch a non-MPI program (hostname) across multiple nodes.

# Function to run the test
run_test() {
  # Command to be executed on multiple nodes
  COMMAND=$1

  # Number of nodes to use
  NODES=$2

  # Run the command using mpirun
  OUTPUT=$(mpirun -np $NODES $COMMAND)

  # Print the output
  echo "Output of mpirun -np $NODES $COMMAND:"
  echo "$OUTPUT"

  # Verify that the command was executed on the specified number of nodes
  NODE_COUNT=$(echo "$OUTPUT" | wc -l)

  if [ "$NODE_COUNT" -eq "$NODES" ]; then
    echo "Test passed: Command executed on $NODE_COUNT nodes."
  else
    echo "Test failed: Expected $NODES nodes, but got $NODE_COUNT nodes."
  fi
}

# Example usage
# Run hostname on 4 nodes
run_test "hostname" 4

# Run uptime on 3 nodes
run_test "uptime" 3
