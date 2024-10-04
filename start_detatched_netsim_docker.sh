# Prior to development do the following:
#docker pull adammwea/netpyneshifter:v5
docker pull adammwea/benshalom_netpyne:v1.0

# Stop and remove the existing container (if it exists)
docker rm -f netsim_docker 2>/dev/null || true

# Run Docker indefinitely detached with the name 'netsim_docker'
docker run -d --rm \
  --name netsim_docker \
  --gpus all \
  -v $(pwd)/../RBS_network_simulations:/app \
  -v /mnt/ben-shalom_nas:/data \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /app \
  adammwea/benshalom_netpyne:v1.0 \
  tail -f /dev/null
  #bash -c 'export PYTHONPATH=/app/submodules/netpyne:$PYTHONPATH >> ~/.bashrc; source ~/.bashrc; tail -f /dev/null'
  

# Check the path and python path
#docker exec netsim_docker bash -c 'export PYTHONPATH=/app/submodules/netpyne:$PYTHONPATH >> ~/.bashrc; source ~/.bashrc; echo $PYTHONPATH'
# docker exec netsim_docker bash -c 'echo $PATH'
# docker exec netsim_docker bash -c 'echo $PYTHONPATH'