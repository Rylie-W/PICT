#!/bin/bash

# Script to run 128 turbulence experiments
# Usage: ./run_128_experiments.sh [dt] [num_experiments]

# Default parameters
DT=${1:-"auto"}  # Use "auto" for CFL-based calculation, or specify custom dt
NUM_EXPERIMENTS=${2:-128}
SAVE_DIR="/mnt/data/yiwei/training_data/turbulence"

echo "=========================================="
echo "Running 128 Turbulence Experiments"
echo "=========================================="
echo "Time step: $DT"
echo "Number of experiments: $NUM_EXPERIMENTS"
echo "Save directory: $SAVE_DIR"
echo "=========================================="

# Create save directory
mkdir -p "$SAVE_DIR"

# Run the experiments
if [ "$DT" = "auto" ]; then
    echo "Using automatic CFL-based time step calculation"
    python generate_128_experiments.py \
        --num_experiments $NUM_EXPERIMENTS \
        --save_dir "$SAVE_DIR" \
        --viscosity 1e-3 \
        --max_velocity 4.2 \
        --domain_scale 1.0 \
        --start_time 4.5 \
        --end_time 25.0 \
        --num_samples 166
else
    echo "Using custom time step: $DT"
    python generate_128_experiments.py \
        --num_experiments $NUM_EXPERIMENTS \
        --save_dir "$SAVE_DIR" \
        --dt $DT \
        --viscosity 1e-3 \
        --max_velocity 4.2 \
        --domain_scale 1.0 \
        --start_time 4.5 \
        --end_time 25.0 \
        --num_samples 166
fi

echo "=========================================="
echo "Experiments completed!"
echo "Results saved to: $SAVE_DIR"
echo "=========================================="

