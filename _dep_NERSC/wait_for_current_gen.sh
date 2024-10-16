current_gen=$1

#if current_get is < 0, set to 0
if ((current_gen < 0)); then
    current_gen=0
fi
echo "current_gen:" $current_gen
# Wait for commands in commands.txt to contain commands for the current generation, use inotifywait
while true; do
    inotifywait -e modify commands.txt
    if grep -q -E ".*gen_${current_gen}.*" NERSC/commands.txt; then
        echo "Found commands for current generation in commands.txt"
        break
    else
        echo "Waiting for commands for current generation..."
    fi
done
echo "test completed successfully"