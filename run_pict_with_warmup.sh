#!/bin/bash
set -e
set -x

# PICT Decaying turbulence data generation using warmup data initialization
# This script demonstrates the new warmup data feature for comparison with reference data

echo "Starting PICT simulation with warmup data initialization..."

python generate_turbulence_data_pict.py \
    --use_warmup_data_init \
    --warmup_segment 1 \
    --enable_comparison \
    --training_data_dir "./training_data/warmup_data" \
    --high_res 128 \
    --low_res 64 \
    --save_interval 1 \
    --generate_steps 5 \
    --warmup_time 4.0 \
    --max_velocity 4.2 \
    --peak_wavenumber 4 \
    --decay \
    --save_file "pict_from_warmup_with_comparison" \
    --save_index 1 \
    --seed 42 \
    --dims 2 \
    --save_dir "./data/pict_warmup_comparison"

echo "PICT simulation with warmup data and comparison completed!"
echo "Generated data can be found in ./data/pict_warmup_comparison/"
echo ""
echo "Comparison results include:"
echo "  - Step-by-step velocity field comparisons: ./data/pict_warmup_comparison/comparison_step_*/"
echo "  - Error evolution plots: ./data/pict_warmup_comparison/comparison_summary/"
echo "  - Detailed statistics for each step"
echo ""
echo "These files allow detailed analysis of PICT simulation quality vs reference data."
