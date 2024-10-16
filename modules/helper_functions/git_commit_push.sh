#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]; then
    echo "Usage: $0 -m \"commit message\""
    exit 1
fi

# Parse command-line arguments
while getopts ":m:" opt; do
  case $opt in
    m)
      commit_message=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Function to add, commit, and push changes in the current repository
commit_and_push() {
    repo_path="$1"
    
    cd "$repo_path" || exit 1

    echo "Processing repository: $(pwd)"

    # Add all changes
    git add .

    # Check if there are any changes to commit
    if [[ -n $(git status --porcelain) ]]; then
        # Commit changes with the provided message
        git commit -m "$commit_message"

        # Push changes to the current branch
        git push
    else
        echo "No changes to commit in $(pwd)"
    fi
}

# Start at the root of the main repository
main_repo=$(git rev-parse --show-toplevel)

# Commit and push changes in the main repo
commit_and_push "$main_repo"

# Loop through submodules and commit/push changes directly without function call
echo "Processing submodules..."
git submodule foreach --recursive '
    echo "Processing submodule: $(pwd)"
    git add .
    if [[ -n $(git status --porcelain) ]]; then
        git commit -m "'"$commit_message"'"
        git push
    else
        echo "No changes to commit in $(pwd)"
    fi
'

echo "All repositories processed successfully!"