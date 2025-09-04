#!/bin/bash

# 生成2048分辨率湍流演化视频的脚本
# 作者: PICT项目
# 用途: 自动化生成高分辨率湍流演化视频

# 设置脚本在遇到错误时退出
set -e

echo "=========================================="
echo "PICT - 湍流演化视频生成脚本"
echo "分辨率: 2048x2048"
echo "结束步数: 12000"
echo "=========================================="

# 配置参数
BASE_PATH="/Volumes/T7/2048"  # 请根据实际数据路径修改
START_STEP=1000
END_STEP=12000
TIME_INTERVAL=1  # 每10帧取1帧，可以调整以控制视频长度
FPS=60  # 视频帧率

# 输出文件配置
OUTPUT_DIR="./output_videos"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "配置参数:"
echo "  数据路径: $BASE_PATH"
echo "  起始步数: $START_STEP"
echo "  结束步数: $END_STEP"
echo "  时间间隔: $TIME_INTERVAL"
echo "  视频帧率: $FPS"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查数据路径是否存在
if [ ! -d "$BASE_PATH" ]; then
    echo "错误: 数据路径不存在: $BASE_PATH"
    echo "请修改脚本中的BASE_PATH变量为正确的数据路径"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "create_turbulence_video.py" ]; then
    echo "错误: create_turbulence_video.py 文件不存在"
    echo "请确保在PICT项目根目录下运行此脚本"
    exit 1
fi

# 生成速度幅值视频
echo "生成速度幅值视频..."
VELOCITY_OUTPUT="$OUTPUT_DIR/turbulence_velocity_magnitude_2048_${TIMESTAMP}.mp4"

python create_turbulence_video.py \
    --base_path "$BASE_PATH" \
    --start_step $START_STEP \
    --end_step $END_STEP \
    --time_interval $TIME_INTERVAL \
    --output_file "$VELOCITY_OUTPUT" \
    --visualization_type velocity_magnitude \
    --fps $FPS

if [ $? -eq 0 ]; then
    echo "✓ 速度幅值视频生成成功: $VELOCITY_OUTPUT"
else
    echo "✗ 速度幅值视频生成失败"
    exit 1
fi

echo ""

# 生成涡度视频
echo "生成涡度视频..."
VORTICITY_OUTPUT="$OUTPUT_DIR/turbulence_vorticity_2048_${TIMESTAMP}.mp4"

python create_turbulence_video.py \
    --base_path "$BASE_PATH" \
    --start_step $START_STEP \
    --end_step $END_STEP \
    --time_interval $TIME_INTERVAL \
    --output_file "$VORTICITY_OUTPUT" \
    --visualization_type vorticity \
    --fps $FPS \
    --vmin -3.0 \
    --vmax 3.0

if [ $? -eq 0 ]; then
    echo "✓ 涡度视频生成成功: $VORTICITY_OUTPUT"
else
    echo "✗ 涡度视频生成失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "视频生成完成！"
echo ""
echo "生成的文件:"
echo "  速度幅值视频: $VELOCITY_OUTPUT"
echo "  涡度视频: $VORTICITY_OUTPUT"
echo ""

# 显示文件大小信息
if [ -f "$VELOCITY_OUTPUT" ]; then
    VELOCITY_SIZE=$(du -h "$VELOCITY_OUTPUT" | cut -f1)
    echo "  速度幅值视频大小: $VELOCITY_SIZE"
fi

if [ -f "$VORTICITY_OUTPUT" ]; then
    VORTICITY_SIZE=$(du -h "$VORTICITY_OUTPUT" | cut -f1)
    echo "  涡度视频大小: $VORTICITY_SIZE"
fi

echo ""
echo "提示："
echo "  - 如需调整视频长度，请修改脚本中的TIME_INTERVAL参数"
echo "  - 如需调整视频帧率，请修改脚本中的FPS参数"
echo "  - 如需修改数据路径，请更新脚本中的BASE_PATH变量"
echo "=========================================="
