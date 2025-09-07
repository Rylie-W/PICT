#!/bin/bash
set -e
set -x

export CUDA_VISIBLE_DEVICES=0

python generate_turbulence_data_pict.py \
    --use_warmup_data_init \
    --training_data_dir "./training_data" \
    --high_res 256 \
    --low_res 256 \
    --save_interval 10 \
    --generate_steps 34770 \
    --warmup_time 40.0 \
    --max_velocity 7.0 \
    --viscosity 1e-3 \
    --peak_wavenumber 2 \
    --cfl_safety_factor 0.5 \
    --integral_scale_factor 3.0 \
    --taylor_reynolds 20.0 \
    --kolmogorov \
    --forcing_scale 1.0 \
    --linear_coefficient -0.1 \
    --decay \
    --save_file "turbulence" \
    --save_index 1 \
    --seed 42 \
    --dims 2 \
    --save_dir "/mnt/data/yiwei/kf/512" \
    --visualize_max_steps 2 \
    --downsample_method "area_average" \
    --warmup_res 256
