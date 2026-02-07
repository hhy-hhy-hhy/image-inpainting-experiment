import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

def load_results(results_path):
    """加载实验结果"""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results

def plot_convergence_curves(results, image_name='ppt3', ratio=0.5):
    """绘制收敛曲线"""
    if image_name not in results:
        print(f"Image {image_name} not found in results")
        return
    
    if ratio not in results[image_name]:
        print(f"Ratio {ratio} not found for image {image_name}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 提取收敛数据
    for algo_name, algo_results in results[image_name][ratio].items():
        info = algo_results['info']
        
        if 'objective_values' in info:
            obj_vals = info['objective_values']
            iterations = range(1, len(obj_vals) + 1)
            
            axes[0].plot(iterations, obj_vals, label=algo_name, linewidth=2)
        
        if 'psnr_values' in info and info['psnr_values'] is not None:
            psnr_vals = info['psnr_values']
            if len(psnr_vals) > 0:
                iterations = range(1, len(psnr_vals) + 1)
                axes[1].plot(iterations, psnr_vals, label=algo_name, linewidth=2)
    
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Objective Function Value')
    axes[0].set_title('Convergence of Objective Function')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('PSNR vs Iteration')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Convergence Analysis: {image_name}, Missing Ratio: {ratio*100:.0f}%')
    plt.tight_layout()
    plt.savefig(f'results/convergence/convergence_{image_name}_ratio{int(ratio*100)}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_comparison(results):
    """绘制性能对比图"""
    # 从汇总报告中读取数据
    summary_path = 'results/experiment_1/summary_report.json'
    
    if not os.path.exists(summary_path):
        print("Summary report not found. Please run the experiment first.")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # 提取数据
    algorithms = summary['algorithms']
    missing_ratios = summary['missing_ratios']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PSNR对比
    for algo_name in algorithms:
        psnr_values = []
        for ratio in missing_ratios:
            if str(ratio) in summary['averages'] and algo_name in summary['averages'][str(ratio)]:
                psnr_values.append(summary['averages'][str(ratio)][algo_name]['avg_psnr'])
        
        if psnr_values:
            axes[0, 0].plot(missing_ratios, psnr_values, 'o-', linewidth=2, 
                           label=algo_name, markersize=8)
    
    axes[0, 0].set_xlabel('Missing Ratio')
    axes[0, 0].set_ylabel('Average PSNR (dB)')
    axes[0, 0].set_title('PSNR vs Missing Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # SSIM对比
    for algo_name in algorithms:
        ssim_values = []
        for ratio in missing_ratios:
            if str(ratio) in summary['averages'] and algo_name in summary['averages'][str(ratio)]:
                ssim_values.append(summary['averages'][str(ratio)][algo_name]['avg_ssim'])
        
        if ssim_values:
            axes[0, 1].plot(missing_ratios, ssim_values, 'o-', linewidth=2, 
                           label=algo_name, markersize=8)
    
    axes[0, 1].set_xlabel('Missing Ratio')
    axes[0, 1].set_ylabel('Average SSIM')
    axes[0, 1].set_title('SSIM vs Missing Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 时间对比
    for algo_name in algorithms:
        time_values = []
        for ratio in missing_ratios:
            if str(ratio) in summary['averages'] and algo_name in summary['averages'][str(ratio)]:
                time_values.append(summary['averages'][str(ratio)][algo_name]['avg_time'])
        
        if time_values:
            axes[1, 0].plot(missing_ratios, time_values, 'o-', linewidth=2, 
                           label=algo_name, markersize=8)
    
    axes[1, 0].set_xlabel('Missing Ratio')
    axes[1, 0].set_ylabel('Average Time (s)')
    axes[1, 0].set_title('Computation Time vs Missing Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 柱状图：在50%缺失比例下的性能对比
    ratio = 0.5
    if str(ratio) in summary['averages']:
        algo_names = []
        psnr_values = []
        
        for algo_name in algorithms:
            if algo_name in summary['averages'][str(ratio)]:
                algo_names.append(algo_name)
                psnr_values.append(summary['averages'][str(ratio)][algo_name]['avg_psnr'])
        
        bars = axes[1, 1].bar(algo_names, psnr_values, color=['blue', 'green', 'red', 'orange'])
        axes[1, 1].set_xlabel('Algorithm')
        axes[1, 1].set_ylabel('Average PSNR (dB)')
        axes[1, 1].set_title(f'PSNR Comparison (Missing Ratio: {ratio*100:.0f}%)')
        
        # 在柱状图上添加数值
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.2f}', ha='center', va='bottom')
    
    plt.suptitle('Algorithm Performance Comparison on Set14 Dataset', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 创建目录
    Path('results/convergence').mkdir(parents=True, exist_ok=True)
    
    # 加载结果
    results_path = 'results/experiment_1/experiment_results.pkl'
    
    if not os.path.exists(results_path):
        print("Experiment results not found. Please run the experiment first.")
        return
    
    results = load_results(results_path)
    
    # 生成图表
    print("Generating convergence curves...")
    plot_convergence_curves(results, 'ppt3', 0.5)
    
    print("Generating performance comparison charts...")
    plot_performance_comparison(results)
    
    print("All charts generated successfully!")

if __name__ == "__main__":
    main()
