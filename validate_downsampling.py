#!/usr/bin/env python3
"""
验证downsample实现正确性的脚本
对比 downsample_checking 目录中的 check 文件和 ref 文件
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def examine_npz_structure(filepath):
    """检查npz文件的结构"""
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"文件: {os.path.basename(filepath)}")
        print(f"字段: {list(data.keys())}")
        for key in data.keys():
            field_data = data[key]
            if hasattr(field_data, 'shape'):
                print(f"  {key}: shape={field_data.shape}, dtype={field_data.dtype}")
            else:
                print(f"  {key}: {type(field_data)}")
        print("-" * 50)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compare_files(check_file, ref_file):
    """对比check文件和ref文件"""
    print(f"对比文件: {os.path.basename(check_file)} vs {os.path.basename(ref_file)}")
    
    try:
        check_data = np.load(check_file, allow_pickle=True)
        ref_data = np.load(ref_file, allow_pickle=True)
        
        # 检查字段是否匹配
        check_keys = set(check_data.keys())
        ref_keys = set(ref_data.keys())
        
        if check_keys != ref_keys:
            print(f"警告: 字段不匹配!")
            print(f"  check独有: {check_keys - ref_keys}")
            print(f"  ref独有: {ref_keys - check_keys}")
        
        results = {}
        common_keys = check_keys & ref_keys
        
        for key in common_keys:
            check_field = check_data[key]
            ref_field = ref_data[key]
            
            # 如果是数组，计算差异
            if hasattr(check_field, 'shape') and hasattr(ref_field, 'shape'):
                if check_field.shape == ref_field.shape:
                    # 特殊处理布尔类型
                    if check_field.dtype == bool and ref_field.dtype == bool:
                        # 布尔类型比较
                        matches = np.sum(check_field == ref_field)
                        total = check_field.size
                        mismatch_ratio = (total - matches) / total
                        
                        results[key] = {
                            'type': 'boolean',
                            'total_elements': total,
                            'matches': matches,
                            'mismatches': total - matches,
                            'mismatch_ratio': mismatch_ratio,
                            'shape': check_field.shape,
                            'all_match': mismatch_ratio == 0
                        }
                    else:
                        # 数值类型比较
                        diff = np.abs(check_field - ref_field)
                        max_diff = np.max(diff)
                        mean_diff = np.mean(diff)
                        rel_error = np.max(diff / (np.abs(ref_field) + 1e-10))
                        
                        results[key] = {
                            'type': 'numeric',
                            'max_abs_diff': max_diff,
                            'mean_abs_diff': mean_diff,
                            'max_rel_error': rel_error,
                            'shape': check_field.shape,
                            'check_range': (np.min(check_field), np.max(check_field)),
                            'ref_range': (np.min(ref_field), np.max(ref_field))
                        }
                else:
                    results[key] = {
                        'error': f'形状不匹配: check={check_field.shape}, ref={ref_field.shape}'
                    }
            else:
                # 非数组数据的比较
                if np.array_equal(check_field, ref_field):
                    results[key] = {'status': '完全相同'}
                else:
                    results[key] = {'status': '不同', 'check': check_field, 'ref': ref_field}
        
        return results
        
    except Exception as e:
        print(f"Error comparing files: {e}")
        return None

def analyze_all_files(downsample_dir):
    """分析所有文件对"""
    downsample_path = Path(downsample_dir)
    
    # 获取所有check和ref文件
    check_files = sorted(list(downsample_path.glob('*check*.npz')))
    ref_files = sorted(list(downsample_path.glob('*ref*.npz')))
    
    print(f"找到 {len(check_files)} 个check文件和 {len(ref_files)} 个ref文件")
    
    # 匹配文件对
    file_pairs = []
    for check_file in check_files:
        check_name = check_file.name.replace('_check_', '_ref_')
        ref_file = downsample_path / check_name
        if ref_file.exists():
            file_pairs.append((check_file, ref_file))
        else:
            print(f"警告: 找不到对应的ref文件: {check_name}")
    
    print(f"匹配到 {len(file_pairs)} 个文件对")
    
    # 分析第一个文件的结构
    if file_pairs:
        print("\n=== 数据结构分析 ===")
        examine_npz_structure(file_pairs[0][0])  # check文件
        examine_npz_structure(file_pairs[0][1])  # ref文件
    
    # 对比所有文件对
    all_results = {}
    print("\n=== 文件对比分析 ===")
    
    for i, (check_file, ref_file) in enumerate(file_pairs):
        print(f"\n对比 {i+1}/{len(file_pairs)}")
        results = compare_files(check_file, ref_file)
        if results:
            all_results[check_file.name] = results
    
    return all_results

def generate_report(results, output_file=None):
    """生成分析报告"""
    if not results:
        print("没有结果可以报告")
        return
    
    report_lines = []
    report_lines.append("=== DOWNSAMPLE验证报告 ===\n")
    
    # 汇总统计
    total_files = len(results)
    all_fields = set()
    for file_results in results.values():
        all_fields.update(file_results.keys())
    
    report_lines.append(f"总文件对数: {total_files}")
    report_lines.append(f"检测到的字段: {sorted(all_fields)}\n")
    
    # 每个字段的汇总统计
    for field in sorted(all_fields):
        field_stats = []
        max_diffs = []
        rel_errors = []
        
        for filename, file_results in results.items():
            if field in file_results:
                if file_results[field].get('type') == 'numeric' and 'max_abs_diff' in file_results[field]:
                    max_diffs.append(file_results[field]['max_abs_diff'])
                    rel_errors.append(file_results[field]['max_rel_error'])
        
        if max_diffs:
            report_lines.append(f"字段 '{field}':")
            report_lines.append(f"  最大绝对差异: {np.max(max_diffs):.2e} (平均: {np.mean(max_diffs):.2e})")
            report_lines.append(f"  最大相对误差: {np.max(rel_errors):.2e} (平均: {np.mean(rel_errors):.2e})")
            report_lines.append("")
    
    # 详细的每文件报告
    report_lines.append("\n=== 详细文件报告 ===")
    for filename, file_results in results.items():
        report_lines.append(f"\n文件: {filename}")
        for field, stats in file_results.items():
            if 'error' in stats:
                report_lines.append(f"  {field}: {stats['error']}")
            elif stats.get('type') == 'numeric':
                report_lines.append(f"  {field} (数值):")
                report_lines.append(f"    最大绝对差异: {stats['max_abs_diff']:.2e}")
                report_lines.append(f"    平均绝对差异: {stats['mean_abs_diff']:.2e}")
                report_lines.append(f"    最大相对误差: {stats['max_rel_error']:.2e}")
                report_lines.append(f"    数据形状: {stats['shape']}")
            elif stats.get('type') == 'boolean':
                report_lines.append(f"  {field} (布尔):")
                report_lines.append(f"    匹配元素: {stats['matches']}/{stats['total_elements']}")
                report_lines.append(f"    不匹配比例: {stats['mismatch_ratio']:.2e}")
                report_lines.append(f"    完全匹配: {stats['all_match']}")
                report_lines.append(f"    数据形状: {stats['shape']}")
            elif 'status' in stats:
                report_lines.append(f"  {field}: {stats['status']}")
    
    report_text = '\n'.join(report_lines)
    print(report_text)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n报告已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='验证downsample实现正确性')
    parser.add_argument('--dir', '-d', 
                        default='/Users/yiwei/Projects/Python/thesis/PICT/downsample_checking',
                        help='包含check和ref文件的目录')
    parser.add_argument('--output', '-o', 
                        default='downsample_validation_report.txt',
                        help='输出报告文件名')
    parser.add_argument('--examine-only', action='store_true',
                        help='只检查文件结构，不进行对比')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"错误: 目录不存在: {args.dir}")
        return
    
    if args.examine_only:
        # 只检查文件结构
        check_files = list(Path(args.dir).glob('*check*.npz'))
        if check_files:
            examine_npz_structure(check_files[0])
    else:
        # 进行完整分析
        results = analyze_all_files(args.dir)
        generate_report(results, args.output)

if __name__ == '__main__':
    main()
