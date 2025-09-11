export CUDA_VISIBLE_DEVICES=1
# 计算缩放参数
# 分辨率比例: 1024/64 = 16
RESOLUTION_RATIO=16

# 保持相同的积分长度尺度在物理空间中的比例
# 64分辨率时: integral_scale_factor=3.0 → L_integral=21.3像素
# 1024分辨率时: 为了保持相同比例，L_integral应该≈340像素
# 所以 integral_scale_factor = 1024/340 ≈ 3.0 (保持不变)

python generate_turbulence_data_pict.py \
    --training_data_dir "/mnt/data/yiwei/pict/64" \
    --ref_data_prefix "turbulence_ref" \
    --downsample_start_step 0 \
    --downsample_end_step 7000 \
    --high_res 64 \
    --low_res 64 \
    --warmup_res 128 \
    --save_interval 10 \
    --generate_steps 12200 \
    --warmup_time 4.0 \
    --max_velocity 4.2 \
    --viscosity 1e-3 \
    --peak_wavenumber 2 \
    --cfl_safety_factor 0.3 \
    --integral_scale_factor 3.0 \
    --taylor_reynolds 20.0 \
    --decay \
    --save_file "turbulence_check" \
    --save_index 1 \
    --seed 42 \
    --dims 2 \
    --save_dir "/mnt/data/yiwei/pict/64" \
    --visualize_max_steps 2 \
    --warmup_timestep 0.011687472669604885 \
    --training_timestep 0.011687472669604885