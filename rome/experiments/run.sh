#!/bin/bash

# Set the command to run
COMMAND_TO_RUN="python3 evaluate_neighbour.py --model=gpt2-xl --alg_name=ROME --hparams_fname=/home/hthakur/model_editing/rome/hparams/ROME/gpt2-xl.json --skip_generation_tests"

# Set the delay between retries (in seconds)
DELAY_SECONDS=5

while true; do
    # Run the command
    $COMMAND_TO_RUN

    # Capture the exit status of the last command
    exit_status=$?

    # Check if the process exited normally (status 0)
    if [ $exit_status -eq 0 ]; then
        echo "Process exited successfully."
        break
    else
        echo "Process exited with status $exit_status. Restarting in $DELAY_SECONDS seconds..."
        sleep $DELAY_SECONDS
    fi
done