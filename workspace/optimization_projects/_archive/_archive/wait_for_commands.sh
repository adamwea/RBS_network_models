gen_pop=$1
count=0
timeout=15
start_time=$(date +%s)

while true; do 
    inotifywait --event modify --timeout $timeout "commands.txt" && {
        count=$(wc -l < "commands.txt")
        echo "line added to commands.txt"
        if (( count >= gen_pop )); then
            break
        fi
    } || {
        current_time=$(date +%s)
        if (( current_time - start_time >= timeout )); then
            echo "Timeout reached without modification"
            count=$(wc -l < "commands.txt")
            echo "Current count: $count"
            if (( count >= gen_pop )); then
                break
            fi
        fi
    }
done
