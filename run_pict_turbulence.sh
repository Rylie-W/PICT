#!/bin/bash
set -e
set -x

# PICT Decaying turbulence training data generation
# Equivalent to the JAX-CFD command but using PICT

python generate_turbulence_data_pict.py \
    --high_res 2048 \
    --low_res 64 \
    --save_interval 50 \
    --generate_steps 12200 \
    --warmup_time 40.0 \
    --max_velocity 4.2 \
    --peak_wavenumber 4 \
    --decay \
    --save_file "decaying_turbulence" \
    --save_index 1 \
    --seed 42 \
    --dims 2 \
    --save_dir "./data/pict_turbulence" 