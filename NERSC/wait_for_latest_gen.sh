run_path=$1

# get directories in run_path
dirs=$(ls -d ${run_path}/*/)

# of those including 'gen_###', get the one with the highest number
gen_pop=$(echo $dirs | grep -o 'gen_[0-9]*' | sort -n | tail -n 1)

# remove 'gen_' prefix from gen_pop
gen_pop=${gen_pop#gen_}


timeout=15
start_time=$(date +%s)

while true; do 
    inotifywait --event modify --timeout $timeout "commands.txt" && {
        latest_gen=$(grep -o 'gen_[0-9]*' "commands.txt" | sort -n | tail -n 1)
        echo "Latest generation: $latest_gen"
        if [[ "$latest_gen" == "gen_$gen_pop" ]]; then
            break
        fi
    } || {
        current_time=$(date +%s)
        if (( current_time - start_time >= timeout )); then
            echo "Timeout reached without modification"
            latest_gen=$(grep -o 'gen_[0-9]*' "commands.txt" | sort -n | tail -n 1)
            echo "Latest generation: $latest_gen"
            if [[ "$latest_gen" == "gen_$gen_pop" ]]; then
                break
            fi
        fi
    }
done


