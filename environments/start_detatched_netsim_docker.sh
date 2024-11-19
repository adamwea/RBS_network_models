#!/bin/bash
cd workspace/RBS_network_simulations_v2/

# Use git to get the root directory of the repo
echo "Determining the root directory of the repository..."
root_dir=$(git rev-parse --show-toplevel)
echo "Root directory: $root_dir"

# Pull the Docker image
echo "Pulling the Docker image adammwea/benshalom_netpyne:v1.0..."
docker pull adammwea/benshalom_netpyne:v1.0

# Stop and remove the existing container (if it exists)
echo "Stopping and removing any existing container named 'netsim_docker'..."
docker rm -f netsim_docker 2>/dev/null || true

# Run Docker indefinitely detached with the name 'netsim_docker'
echo "Running Docker container 'netsim_docker' in detached mode..."
docker run -d --rm \
  --name netsim_docker \
  -v "$root_dir:/app" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /app \
  adammwea/benshalom_netpyne:v1.0 \
  tail -f /dev/null

echo "Docker container 'netsim_docker' is now running."
  
# # Run Docker indefinitely detached with the name 'netsim_docker'
# docker run -d --rm \
#   --name netsim_docker \
#   --gpus all \
#   -v $(pwd)/../RBS_network_simulations:/app \
#   -v /mnt/ben-shalom_nas:/data \
#   -v /var/run/docker.sock:/var/run/docker.sock \
#   -w /app \
#   adammwea/benshalom_netpyne:v1.0 \
#   tail -f /dev/null
#   #bash -c 'export PYTHONPATH=/app/submodules/netpyne:$PYTHONPATH >> ~/.bashrc; source ~/.bashrc; tail -f /dev/null'
  

# Check the path and python path
#docker exec netsim_docker bash -c 'export PYTHONPATH=/app/submodules/netpyne:$PYTHONPATH >> ~/.bashrc; source ~/.bashrc; echo $PYTHONPATH'
# docker exec netsim_docker bash -c 'echo $PATH'
# docker exec netsim_docker bash -c 'echo $PYTHONPATH'