#!/bin/bash
# run_inference_and_generate.sh
#
# Usage: ./run_inference_and_generate.sh <dataset> <split> <run_split>
#
# This script loops over checkpoints 1 to 10, starting the inference server
# in a new process group, generating logits, and then killing the entire
# inference process group for each checkpoint.
#
# It uses `setsid` to start the inference process in its own session (and process group),
# then retrieves the PGID and later kills the whole group with `kill -TERM -<PGID>`.
#
# Note: Make sure your system has `setsid` available (commonly found on Linux).

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <dataset> <split> <run_split>"
    exit 1
fi

DATASET=$1
SPLIT=$2
RUN_SPLIT=$3

for checkpoint in {0..15}; do
    echo "============================"
    echo "Processing checkpoint: $checkpoint"
    echo "============================"

    # Start the inference server in a new process group using setsid.
    setsid bash ./shell/start_inference.sh "$DATASET" "$SPLIT" "$checkpoint" &
    INFERENCE_PID=$!

    # Retrieve the process group ID (PGID) for the inference process.
    PGID=$(ps -o pgid= -p "$INFERENCE_PID" | tr -d ' ')
    echo "Started inference (PID: $INFERENCE_PID, PGID: $PGID). Waiting for the server to initialize..."
    
    # Wait for the server to fully initialize.
    sleep 120

    # Generate logits.
    echo "Generating logits for checkpoint $checkpoint..."
    bash ./shell/start_logit_gen.sh "$checkpoint" "$DATASET" "$SPLIT" "$RUN_SPLIT"
    
    # Kill the entire inference process group.
    echo "Killing inference process group (PGID: $PGID)..."
    kill -TERM -"$PGID"
    
    # Wait for the main inference process to exit cleanly.
    wait "$INFERENCE_PID" 2>/dev/null
    echo "Checkpoint $checkpoint processed."
    
    # Brief pause before moving to the next checkpoint.
    sleep 20
done

echo "All checkpoints processed successfully."
