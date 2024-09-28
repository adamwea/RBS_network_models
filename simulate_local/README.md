# Prior to development do the following:
docker pull adammwea/netpyneshifter:v5

# Run Docker indefinitely detatched
docker run -d --rm \
  -v /home/adamm/workspace/network_simulations:/app \
  -v /mnt/ben-shalom_nas:/data \
  -w /app adammwea/netpyneshifter:v5 \
  tail -f /dev/null

# Attatch VS code instance to it - this may take a few minutes to start up
Press F1 (or Cmd+Shift+P on macOS) to open the Command Palette in VS Code.
Dev-Container > Attatch to running container

# Navigate to /app if needed...but hopefully .devcontainer will work for that next time.

