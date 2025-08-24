#!/bin/bash
set -e
set -x

# PICT 1024分辨率湍流生成脚本
# 针对高分辨率优化的参数设置

echo "生成1024×1024分辨率湍流场..."
echo "关键调整:"
echo "  - 分辨率: 64×64 → 1024×1024 (16倍)"
echo "  - 保持积分长度尺度: ~21像素 → ~340像素"
echo "  - 时间步长: 自动按CFL条件缩放"
echo "  - 减少warmup和生成步数以节省计算时间"
echo "  - GPU内存需求: ~256倍增加"

# 计算缩放参数
# 分辨率比例: 1024/64 = 16
RESOLUTION_RATIO=16

# 保持相同的积分长度尺度在物理空间中的比例
# 64分辨率时: integral_scale_factor=3.0 → L_integral=21.3像素
# 1024分辨率时: 为了保持相同比例，L_integral应该≈340像素
# 所以 integral_scale_factor = 1024/340 ≈ 3.0 (保持不变)

python generate_turbulence_data_pict.py \
    --high_res 1024 \
    --low_res 1024 \
    --save_interval 1 \
    --generate_steps 12200 \
    --warmup_time 0.0 \
    --max_velocity 4.2 \
    --viscosity 1e-3 \
    --peak_wavenumber 2 \
    --cfl_safety_factor 0.3 \
    --integral_scale_factor 3.0 \
    --taylor_reynolds 20.0 \
    --decay \
    --save_file "turbulence_1024" \
    --save_index 1 \
    --seed 42 \
    --dims 2 \
    --save_dir "./data/turbulence_1024" \
    --visualize_max_steps 2

echo "1024分辨率湍流场生成完成！"
echo "生成的数据位于: ./data/turbulence_1024/"
echo ""
echo "注意事项:"
echo "  - 内存使用量: ~256倍于64分辨率"
echo "  - 计算时间: 显著增加"
echo "  - 文件大小: 每个时间步约4GB"
echo "  - 建议监控GPU内存使用"
