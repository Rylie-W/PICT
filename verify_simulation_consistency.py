#!/usr/bin/env python3
"""
验证从任意步数开始模拟所得的数据与参考数据的一致性

验证逻辑：
- ref数据：从随机初始速度开始的完整模拟
- check数据：从ref数据的某个步数开始继续模拟
- 验证：check/step_X/step_Y 应该与 ref/step_(X+Y) 相同
"""

import os
import numpy as np
import argparse
from pathlib import Path
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation_consistency_validation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_velocity_data(file_path):
    """加载速度数据"""
    try:
        if file_path.suffix == '.npz':
            data = np.load(file_path)
            # 尝试不同的可能键名
            possible_keys = ['velocity', 'arr_0', 'arr_1', 'arr_2']
            for key in possible_keys:
                if key in data:
                    velocity_data = data[key]
                    return velocity_data
            
            # 如果没有找到标准键名，返回第一个数组
            if len(data.keys()) > 0:
                first_key = list(data.keys())[0]
                velocity_data = data[first_key]
                return velocity_data
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_velocity_fields(vel1, vel2, tolerance=1e-10):
    """比较两个速度场"""
    if vel1 is None or vel2 is None:
        return False, {"error": "One or both velocity fields are None"}
    
    # 确保形状相同
    if vel1.shape != vel2.shape:
        return False, {"error": f"Shape mismatch: {vel1.shape} vs {vel2.shape}"}
    
    # 计算差异
    diff = np.abs(vel1 - vel2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    relative_error = max_diff / (np.max(np.abs(vel1)) + 1e-16)
    
    # 判断是否相同
    is_identical = max_diff < tolerance
    
    stats = {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "relative_error": relative_error,
        "is_identical": is_identical,
        "tolerance": tolerance
    }
    
    return is_identical, stats

def find_matching_files(ref_dir, check_dir, start_step):
    """找到需要比较的文件对"""
    ref_dir = Path(ref_dir)
    check_dir = Path(check_dir)
    
    matching_pairs = []
    
    # 获取check目录中的所有文件
    check_files = list(check_dir.glob("turbulence_check_64x64_step_*.npz"))
    
    for check_file in check_files:
        # 从文件名中提取步数
        filename = check_file.name
        if 'turbulence_check_64x64_step_' in filename:
            step_str = filename.replace('turbulence_check_64x64_step_', '').replace('.npz', '')
            try:
                check_step = int(step_str)
                # 计算对应的ref步数：start_step + check_step
                ref_step = start_step + check_step
                
                # 查找对应的ref文件
                ref_file = ref_dir / f"turbulence_ref_64x64_step_{ref_step}.npz"
                
                if ref_file.exists():
                    matching_pairs.append({
                        'check_file': check_file,
                        'ref_file': ref_file,
                        'check_step': check_step,
                        'ref_step': ref_step,
                        'start_step': start_step
                    })
            except ValueError:
                continue
    
    return matching_pairs

def validate_consistency(ref_dir, check_dir, logger, max_comparisons=50):
    """验证模拟一致性"""
    ref_dir = Path(ref_dir)
    check_dir = Path(check_dir)
    
    results = []
    
    # 获取所有的step目录
    step_dirs = [d for d in check_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]
    
    logger.info(f"Found {len(step_dirs)} step directories: {[d.name for d in step_dirs]}")
    
    for step_dir in step_dirs:
        start_step_str = step_dir.name.replace('step_', '')
        try:
            start_step = int(start_step_str)
        except ValueError:
            logger.warning(f"Invalid step directory name: {step_dir.name}")
            continue
            
        logger.info(f"\n{'='*60}")
        logger.info(f"验证起始步数: {start_step}")
        logger.info(f"{'='*60}")
        
        # 找到需要比较的文件对
        matching_pairs = find_matching_files(ref_dir, step_dir, start_step)
        
        if not matching_pairs:
            logger.warning(f"No matching files found for start_step {start_step}")
            continue
            
        logger.info(f"Found {len(matching_pairs)} file pairs to compare")
        
        # 限制比较数量以避免过长的运行时间
        if len(matching_pairs) > max_comparisons:
            matching_pairs = matching_pairs[:max_comparisons]
            logger.info(f"Limiting to first {max_comparisons} comparisons")
        
        step_results = {
            'start_step': start_step,
            'total_pairs': len(matching_pairs),
            'identical_pairs': 0,
            'different_pairs': 0,
            'errors': 0,
            'max_difference': 0,
            'comparisons': []
        }
        
        for i, pair in enumerate(matching_pairs):
            if i % 10 == 0:
                logger.info(f"Processing comparison {i+1}/{len(matching_pairs)}")
                
            # 加载数据
            check_vel = load_velocity_data(pair['check_file'])
            ref_vel = load_velocity_data(pair['ref_file'])
            
            # 比较数据
            is_identical, stats = compare_velocity_fields(check_vel, ref_vel)
            
            comparison = {
                'check_step': pair['check_step'],
                'ref_step': pair['ref_step'],
                'is_identical': is_identical,
                'stats': stats
            }
            
            step_results['comparisons'].append(comparison)
            
            if 'error' in stats:
                step_results['errors'] += 1
                logger.error(f"Error comparing step {pair['check_step']} -> {pair['ref_step']}: {stats['error']}")
            elif is_identical:
                step_results['identical_pairs'] += 1
                if stats['max_diff'] > step_results['max_difference']:
                    step_results['max_difference'] = stats['max_diff']
            else:
                step_results['different_pairs'] += 1
                logger.warning(f"Difference found at step {pair['check_step']} -> {pair['ref_step']}: max_diff={stats['max_diff']:.2e}")
                if stats['max_diff'] > step_results['max_difference']:
                    step_results['max_difference'] = stats['max_diff']
        
        results.append(step_results)
        
        # 总结该起始步数的结果
        logger.info(f"\n起始步数 {start_step} 的验证结果:")
        logger.info(f"  总比较数: {step_results['total_pairs']}")
        logger.info(f"  完全相同: {step_results['identical_pairs']}")
        logger.info(f"  存在差异: {step_results['different_pairs']}")
        logger.info(f"  加载错误: {step_results['errors']}")
        logger.info(f"  最大差异: {step_results['max_difference']:.2e}")
        
        if step_results['identical_pairs'] == step_results['total_pairs'] - step_results['errors']:
            logger.info(f"  ✅ 从步数 {start_step} 开始的模拟结果完全一致!")
        else:
            logger.warning(f"  ⚠️  从步数 {start_step} 开始的模拟结果存在差异!")
    
    return results

def generate_report(results, output_file):
    """生成验证报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("模拟一致性验证报告\n")
        f.write("="*80 + "\n\n")
        
        total_comparisons = 0
        total_identical = 0
        total_different = 0
        total_errors = 0
        
        for result in results:
            start_step = result['start_step']
            f.write(f"起始步数: {start_step}\n")
            f.write("-" * 40 + "\n")
            f.write(f"总比较数: {result['total_pairs']}\n")
            f.write(f"完全相同: {result['identical_pairs']}\n")
            f.write(f"存在差异: {result['different_pairs']}\n")
            f.write(f"加载错误: {result['errors']}\n")
            f.write(f"最大差异: {result['max_difference']:.2e}\n")
            
            if result['identical_pairs'] == result['total_pairs'] - result['errors']:
                f.write("状态: ✅ 完全一致\n")
            else:
                f.write("状态: ⚠️ 存在差异\n")
            
            f.write("\n")
            
            total_comparisons += result['total_pairs']
            total_identical += result['identical_pairs']
            total_different += result['different_pairs']
            total_errors += result['errors']
        
        f.write("总体统计\n")
        f.write("="*40 + "\n")
        f.write(f"总比较数: {total_comparisons}\n")
        f.write(f"完全相同: {total_identical}\n")
        f.write(f"存在差异: {total_different}\n")
        f.write(f"加载错误: {total_errors}\n")
        f.write(f"一致性比例: {total_identical/max(total_comparisons-total_errors, 1)*100:.2f}%\n")
        
        if total_identical == total_comparisons - total_errors and total_comparisons > 0:
            f.write("\n🎉 验证结果: 所有从任意步数开始的模拟都与参考数据完全一致!\n")
        else:
            f.write("\n⚠️ 验证结果: 发现了不一致的数据，需要进一步检查。\n")

def main():
    parser = argparse.ArgumentParser(description='验证模拟一致性')
    parser.add_argument('--ref_dir', type=str, 
                       default='/Users/yiwei/Projects/Python/thesis/PICT/downsample_checking/64',
                       help='参考数据目录')
    parser.add_argument('--check_dir', type=str,
                       default='/Users/yiwei/Projects/Python/thesis/PICT/downsample_checking/check',
                       help='检查数据目录')
    parser.add_argument('--max_comparisons', type=int, default=50,
                       help='每个起始步数的最大比较数量')
    parser.add_argument('--tolerance', type=float, default=1e-10,
                       help='数值比较容忍度')
    parser.add_argument('--output', type=str, default='simulation_consistency_report.txt',
                       help='输出报告文件名')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("开始验证模拟一致性...")
    logger.info(f"参考数据目录: {args.ref_dir}")
    logger.info(f"检查数据目录: {args.check_dir}")
    logger.info(f"数值容忍度: {args.tolerance}")
    
    # 验证目录存在
    if not Path(args.ref_dir).exists():
        logger.error(f"参考数据目录不存在: {args.ref_dir}")
        return
        
    if not Path(args.check_dir).exists():
        logger.error(f"检查数据目录不存在: {args.check_dir}")
        return
    
    # 执行验证
    results = validate_consistency(args.ref_dir, args.check_dir, logger, args.max_comparisons)
    
    # 生成报告
    generate_report(results, args.output)
    logger.info(f"验证报告已保存到: {args.output}")

if __name__ == '__main__':
    main()
