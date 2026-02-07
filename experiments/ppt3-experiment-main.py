import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
import pywt
import bm3d
import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 示例：创建包含中文的图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左侧：PSNR收敛曲线
x = np.arange(0, 300, 10)
ax1.plot(x, 20 + 5 * np.sin(x/50), label='ISTA', linewidth=2)
ax1.plot(x, 22 + 5 * np.sin(x/50 + 1), label='FISTA-L1', linewidth=2)
ax1.plot(x, 24 + 5 * np.sin(x/50 + 2), label='FISTA-TV', linewidth=2)
ax1.set_xlabel('迭代次数', fontsize=12)
ax1.set_ylabel('PSNR (dB)', fontsize=12)
ax1.set_title('PSNR收敛曲线', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右侧：SSIM收敛曲线
ax2.plot(x, 0.7 + 0.2 * np.sin(x/50), label='ISTA', linewidth=2)
ax2.plot(x, 0.75 + 0.2 * np.sin(x/50 + 1), label='FISTA-L1', linewidth=2)
ax2.plot(x, 0.8 + 0.2 * np.sin(x/50 + 2), label='FISTA-TV', linewidth=2)
ax2.set_xlabel('迭代次数', fontsize=12)
ax2.set_ylabel('SSIM', fontsize=12)
ax2.set_title('SSIM收敛曲线', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('收敛曲线_中文.png', dpi=300, bbox_inches='tight')
plt.show()
# 设置随机种子
np.random.seed(42)

def load_real_ppt3():
    """加载真实的ppt3.png图像"""
    try:
        # 尝试从不同位置加载
        possible_paths = [
            'ppt3.png',
            './ppt3.png',
            'PPT3.png',
            './PPT3.png'
        ]
        
        image = None
        image_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image_path = path
                    print(f"成功加载图像: {path}")
                    break
        
        if image is None:
            # 创建模拟的ppt3图像
            print("未找到ppt3.png，创建模拟图像")
            image = create_simulated_ppt3()
        else:
            # 调整大小并归一化
            if image.shape[0] > 400 or image.shape[1] > 400:
                image = cv2.resize(image, (256, 256))
            else:
                image = cv2.resize(image, (256, 256))
            
            # 确保是灰度图
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 转换为浮点并归一化
            image = image.astype(np.float32) / 255.0
            
            # 提高对比度
            image = np.clip((image - 0.5) * 1.5 + 0.5, 0, 1)
        
        return image
    
    except Exception as e:
        print(f"加载图像时出错: {e}")
        return create_simulated_ppt3()

def create_simulated_ppt3():
    """创建模拟的ppt3图像"""
    height, width = 256, 256
    
    # 创建白色背景
    image = np.ones((height, width))
    
    # 添加书名文字（模拟书本封面）
    text_lines = [
        "EVERYTHING",
        "HOW TO DO",
        "WITH",
        "POWERPOINT",
        "2002"
    ]
    
    y_positions = [40, 70, 100, 130, 170]
    font_scales = [0.8, 0.7, 0.6, 0.8, 0.6]
    
    for i, (text, y_pos) in enumerate(zip(text_lines, y_positions)):
        font_scale = font_scales[i]
        thickness = 2 if i < 3 else 3
        color = 0.0  # 黑色
        
        # 计算文本大小和位置以居中
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x_pos = (width - text_size[0]) // 2
        
        cv2.putText(image, text, (x_pos, y_pos), font, font_scale, color, thickness)
    
    # 添加一些额外的文本元素
    bullet_points = [
        "• Create effective multimedia",
        "• Make your point meaningfully",
        "• Deliver on-screen or Web"
    ]
    
    for i, point in enumerate(bullet_points):
        y = 200 + i * 25
        cv2.putText(image, point, (30, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0.0, 1)
    
    # 添加作者和出版社
    cv2.putText(image, "Ellen Finkelstein", (30, 230), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0.0, 2)
    cv2.putText(image, "OSBORNE", (width - 100, 230), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0.0, 2)
    
    # 添加一些噪点模拟扫描效果
    image += 0.05 * np.random.randn(height, width)
    return np.clip(image, 0, 1)

def create_corrupted_image(image, missing_rate=0.25):
    """创建有缺失像素的图像"""
    mask = np.random.random(image.shape) > missing_rate
    corrupted = image.copy()
    corrupted[~mask] = 0
    return corrupted, mask

def calculate_image_metrics(original, restored):
    """正确计算图像指标"""
    # 确保数据类型一致且范围正确
    original_clean = np.clip(original.astype(np.float32), 0, 1)
    restored_clean = np.clip(restored.astype(np.float32), 0, 1)
    
    # 计算PSNR
    mse = np.mean((original_clean - restored_clean) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # 计算SSIM
    ssim_val = structural_similarity(original_clean, restored_clean, 
                                    data_range=1.0, win_size=7, 
                                    channel_axis=None)
    
    return psnr, ssim_val

def ista_inpainting(corrupted, mask, original=None, lambda_reg=0.02, max_iter=300):
    """ISTA算法进行图像修复"""
    # 初始化
    x = corrupted.copy()
    
    # 小波变换参数
    wavelet = 'db4'
    level = 3
    
    # 收敛历史记录
    psnr_history = []
    ssim_history = []
    
    # ISTA迭代
    for i in range(max_iter):
        # 小波变换
        coeffs = pywt.wavedec2(x, wavelet, level=level)
        
        # 软阈值处理
        coeffs_thresh = []
        for j, coeff in enumerate(coeffs):
            if j == 0:  # 低频部分不处理
                coeffs_thresh.append(coeff)
            else:
                if isinstance(coeff, tuple):
                    # 处理高频系数
                    coeff_thresh = tuple(soft_threshold(c, lambda_reg) for c in coeff)
                    coeffs_thresh.append(coeff_thresh)
        
        # 逆小波变换
        x_new = pywt.waverec2(coeffs_thresh, wavelet)
        
        # 裁剪到有效范围
        x_new = np.clip(x_new, 0, 1)
        
        # 在已知像素处保持原始值
        x_new[mask] = corrupted[mask]
        
        # 更新
        x = x_new
        
        # 计算指标（使用原始图像作为参考）
        if original is not None and (i % 10 == 0 or i == max_iter - 1):
            psnr, ssim_val = calculate_image_metrics(original, x)
            psnr_history.append(psnr)
            ssim_history.append(ssim_val)
    
    # 如果未记录历史，则计算最终指标
    if len(psnr_history) == 0 and original is not None:
        psnr, ssim_val = calculate_image_metrics(original, x)
        psnr_history.append(psnr)
        ssim_history.append(ssim_val)
    
    return x, psnr_history, ssim_history

def fista_inpainting(corrupted, mask, original=None, lambda_reg=0.02, max_iter=300):
    """FISTA算法进行图像修复（L1正则化）"""
    # 初始化
    x = corrupted.copy()
    y = x.copy()
    t = 1.0
    
    # 小波变换参数
    wavelet = 'db4'
    level = 3
    
    # 收敛历史记录
    psnr_history = []
    ssim_history = []
    
    for i in range(max_iter):
        # 保存旧的x
        x_old = x.copy()
        
        # 小波变换
        coeffs = pywt.wavedec2(y, wavelet, level=level)
        
        # 软阈值处理
        coeffs_thresh = []
        for j, coeff in enumerate(coeffs):
            if j == 0:  # 低频部分不处理
                coeffs_thresh.append(coeff)
            else:
                if isinstance(coeff, tuple):
                    coeff_thresh = tuple(soft_threshold(c, lambda_reg) for c in coeff)
                    coeffs_thresh.append(coeff_thresh)
        
        # 逆小波变换
        x = pywt.waverec2(coeffs_thresh, wavelet)
        
        # 裁剪到有效范围
        x = np.clip(x, 0, 1)
        
        # 在已知像素处保持原始值
        x[mask] = corrupted[mask]
        
        # FISTA加速步骤
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
        
        # 计算指标
        if original is not None and (i % 10 == 0 or i == max_iter - 1):
            psnr, ssim_val = calculate_image_metrics(original, x)
            psnr_history.append(psnr)
            ssim_history.append(ssim_val)
    
    # 如果未记录历史，则计算最终指标
    if len(psnr_history) == 0 and original is not None:
        psnr, ssim_val = calculate_image_metrics(original, x)
        psnr_history.append(psnr)
        ssim_history.append(ssim_val)
    
    return x, psnr_history, ssim_history

def soft_threshold(x, threshold):
    """软阈值函数"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def fista_tv_inpainting(corrupted, mask, original=None, lambda_tv=0.1, max_iter=300):
    """FISTA算法进行图像修复（TV正则化）"""
    x = corrupted.copy()
    y = x.copy()
    t = 1.0
    
    # 步长
    tau = 0.1
    
    # 收敛历史记录
    psnr_history = []
    ssim_history = []
    
    for i in range(max_iter):
        x_old = x.copy()
        
        # 计算梯度
        grad_x = np.roll(y, -1, axis=0) - y
        grad_y = np.roll(y, -1, axis=1) - y
        
        # 计算TV范数的近端算子
        norm = np.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        shrink = np.maximum(norm - lambda_tv, 0) / norm
        
        # 更新
        div_x = np.roll(shrink * grad_x, 1, axis=0) - shrink * grad_x
        div_y = np.roll(shrink * grad_y, 1, axis=1) - shrink * grad_y
        
        x = y - tau * (2 * (y - corrupted) * mask + div_x + div_y)
        
        # 裁剪到有效范围
        x = np.clip(x, 0, 1)
        
        # 在已知像素处保持原始值
        x[mask] = corrupted[mask]
        
        # FISTA加速
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
        
        # 计算指标
        if original is not None and (i % 10 == 0 or i == max_iter - 1):
            psnr, ssim_val = calculate_image_metrics(original, x)
            psnr_history.append(psnr)
            ssim_history.append(ssim_val)
    
    # 如果未记录历史，则计算最终指标
    if len(psnr_history) == 0 and original is not None:
        psnr, ssim_val = calculate_image_metrics(original, x)
        psnr_history.append(psnr)
        ssim_history.append(ssim_val)
    
    return x, psnr_history, ssim_history

def bm3d_inpainting(corrupted, mask, original=None):
    """BM3D算法进行图像修复"""
    # 使用均值填充缺失区域
    init_estimate = corrupted.copy()
    mean_val = np.mean(corrupted[mask])
    init_estimate[~mask] = mean_val
    
    # 转换为0-255范围（BM3D需要）
    init_estimate_255 = (init_estimate * 255).astype(np.float32)
    
    try:
        # 使用BM3D进行修复
        denoised = bm3d.bm3d(init_estimate_255, sigma_psd=20)
    except Exception as e:
        print(f"BM3D算法错误: {e}，使用中值滤波替代")
        denoised = cv2.medianBlur(init_estimate_255.astype(np.uint8), 3)
        denoised = denoised.astype(np.float32)
    
    # 转换回0-1范围
    denoised = denoised.astype(np.float32) / 255.0
    
    # 在已知像素处保持原始值
    denoised[mask] = corrupted[mask]
    
    # 裁剪到有效范围
    denoised = np.clip(denoised, 0, 1)
    
    # 计算指标
    psnr_history = []
    ssim_history = []
    
    if original is not None:
        psnr, ssim_val = calculate_image_metrics(original, denoised)
        psnr_history.append(psnr)
        ssim_history.append(ssim_val)
    
    return denoised, psnr_history, ssim_history
def improved_bm3d_inpainting(corrupted, mask, original=None):
    """
    改进的BM3D图像修复算法
    包含多种改进策略
    """
    
    # 策略1: 邻域加权填充（比简单均值更好）
    def neighborhood_weighted_fill(image, mask, iterations=5):
        filled = image.copy()
        kernel = np.array([[0.5, 1.0, 0.5],
                          [1.0, 0.0, 1.0],
                          [0.5, 1.0, 0.5]]) / 6.0
        
        for _ in range(iterations):
            # 计算加权邻域均值
            weighted = ndimage.convolve(filled, kernel, mode='reflect')
            # 只更新缺失区域
            filled[~mask] = weighted[~mask]
            # 保持已知像素不变
            filled[mask] = image[mask]
        
        return filled
    
    # 策略2: 多尺度修复
    def multi_scale_inpainting(image, mask, scales=[0.5, 0.75, 1.0]):
        results = []
        
        for scale in scales:
            # 缩放图像
            h, w = image.shape
            new_h, new_w = int(h * scale), int(w * scale)
            
            if scale != 1.0:
                img_resized = cv2.resize(image, (new_w, new_h))
                mask_resized = cv2.resize(mask.astype(np.float32), (new_w, new_h)) > 0.5
            else:
                img_resized = image.copy()
                mask_resized = mask.copy()
            
            # 在缩放尺度上填充
            filled = neighborhood_weighted_fill(img_resized, mask_resized)
            
            # 应用BM3D（调整参数）
            filled_255 = (filled * 255).astype(np.float32)
            
            # 根据尺度调整sigma
            sigma = int(25 * scale)
            sigma = max(10, min(sigma, 50))
            
            try:
                denoised = bm3d.bm3d(filled_255, sigma_psd=sigma)
                denoised = denoised.astype(np.float32) / 255.0
                
                # 保持已知像素
                denoised[mask_resized] = img_resized[mask_resized]
                
                if scale != 1.0:
                    # 缩放回原尺寸
                    denoised = cv2.resize(denoised, (w, h))
                
                results.append(denoised)
            except Exception as e:
                print(f"  尺度{scale}失败: {e}")
        
        # 多尺度结果融合
        if results:
            # 简单平均融合
            final_result = np.mean(results, axis=0)
            final_result[mask] = corrupted[mask]  # 确保已知像素不变
            return final_result
        else:
            # 所有尺度都失败，使用中值滤波后备
            print("  所有尺度失败，使用中值滤波后备")
            filled = neighborhood_weighted_fill(corrupted, mask)
            median = cv2.medianBlur((filled * 255).astype(np.uint8), 3)
            return median.astype(np.float32) / 255.0
    
    # 执行多尺度修复
    print("运行改进的BM3D修复算法...")
    result = multi_scale_inpainting(corrupted, mask)
    
    return result
def run_complete_experiment():
    """运行完整的图像修复实验"""
    print("=" * 70)
    print("图像修复实验 - PPT3.PNG")
    print("=" * 70)
    
    # 1. 加载图像
    print("\n1. 加载原始图像...")
    original = load_real_ppt3()
    print(f"   图像尺寸: {original.shape}")
    print(f"   像素范围: [{original.min():.3f}, {original.max():.3f}]")
    
    # 2. 创建损坏图像
    print("\n2. 创建损坏图像...")
    corrupted, mask = create_corrupted_image(original, missing_rate=0.25)
    missing_count = np.sum(~mask)
    total_count = original.size
    print(f"   缺失像素数: {missing_count}/{total_count} ({missing_count/total_count*100:.1f}%)")
    
    # 3. 运行算法
    print("\n3. 运行修复算法...")
    algorithms = {
        'ISTA': (ista_inpainting, {'lambda_reg': 0.02, 'max_iter': 300}),
        'FISTA-L1': (fista_inpainting, {'lambda_reg': 0.02, 'max_iter': 300}),
        'FISTA-TV': (fista_tv_inpainting, {'lambda_tv': 0.08, 'max_iter': 300}),
        'BM3D': (bm3d_inpainting, {})
    }
    
    results = {}
    
    for algo_name, (algo_func, params) in algorithms.items():
        print(f"   - 运行{algo_name}...")
        start_time = time.time()
        
        if algo_name == 'BM3D':
            restored, psnr_history, ssim_history = algo_func(corrupted, mask, original)
        else:
            restored, psnr_history, ssim_history = algo_func(corrupted, mask, original, **params)
        
        elapsed_time = time.time() - start_time
        
        # 计算最终指标
        final_psnr, final_ssim = calculate_image_metrics(original, restored)
        
        results[algo_name] = {
            'restored': restored,
            'psnr': final_psnr,
            'ssim': final_ssim,
            'time': elapsed_time,
            'psnr_history': psnr_history,
            'ssim_history': ssim_history
        }
        
        print(f"     完成: PSNR={final_psnr:.2f} dB, SSIM={final_ssim:.4f}, 时间={elapsed_time:.2f}秒")
    
    # 4. 显示结果
    print("\n" + "=" * 70)
    print("实验结果总结")
    print("=" * 70)
    
    # 创建三线表
    print("\nLaTeX三线表格式:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{图像修复算法性能比较 (PPT3.PNG)}")
    print("\\label{tab:inpainting_results}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("算法 & PSNR (dB) & SSIM & 时间 (s) \\\\")
    print("\\midrule")
    
    for algo_name in algorithms.keys():
        result = results[algo_name]
        psnr_display = "∞" if result['psnr'] == float('inf') else f"{result['psnr']:.2f}"
        print(f"{algo_name} & {psnr_display} & {result['ssim']:.4f} & {result['time']:.2f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # 5. 可视化结果
    print("\n5. 生成可视化结果...")
    plot_all_results(original, corrupted, results)
    
    # 6. 分析报告
    print("\n" + "=" * 70)
    print("算法性能分析")
    print("=" * 70)
    
    # 找出最佳算法
    valid_results = {k: v for k, v in results.items() if v['psnr'] != float('inf')}
    
    if valid_results:
        best_psnr = max(valid_results.items(), key=lambda x: x[1]['psnr'])
        best_ssim = max(valid_results.items(), key=lambda x: x[1]['ssim'])
        fastest = min(results.items(), key=lambda x: x[1]['time'])
        
        print(f"\n1. 最佳PSNR: {best_psnr[0]} ({best_psnr[1]['psnr']:.2f} dB)")
        print(f"2. 最佳SSIM: {best_ssim[0]} ({best_ssim[1]['ssim']:.4f})")
        print(f"3. 最快算法: {fastest[0]} ({fastest[1]['time']:.2f}秒)")
    
    print("\n4. 算法特性分析:")
    print("   - BM3D: 基于块匹配和3D滤波，对纹理恢复效果好")
    print("   - FISTA-L1: 小波域稀疏表示，加速收敛")
    print("   - FISTA-TV: 全变分正则化，边缘保持效果好")
    print("   - ISTA: 基础迭代算法，收敛较慢")
    
    # 7. 保存结果
    print("\n7. 保存结果图像...")
    save_all_images(original, corrupted, results)
    
    return original, corrupted, results

def plot_all_results(original, corrupted, results):
    """绘制所有结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 损坏图像
    axes[0, 1].imshow(corrupted, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('损坏图像 (25%像素缺失)')
    axes[0, 1].axis('off')
    
    # 算法结果
    algorithm_order = ['ISTA', 'FISTA-L1', 'FISTA-TV', 'BM3D']
    positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
    
    for idx, (algo_name, position) in enumerate(zip(algorithm_order, positions)):
        if algo_name in results:
            row, col = position
            axes[row, col].imshow(results[algo_name]['restored'], cmap='gray', vmin=0, vmax=1)
            axes[row, col].set_title(f'{algo_name} 修复结果')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('ppt3_inpainting_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 收敛曲线（仅适用于迭代算法）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'ISTA': 'blue', 'FISTA-L1': 'red', 'FISTA-TV': 'green'}
    
    for algo_name in ['ISTA', 'FISTA-L1', 'FISTA-TV']:
        if algo_name in results and len(results[algo_name]['psnr_history']) > 1:
            iterations = np.arange(len(results[algo_name]['psnr_history'])) * 10
            
            # PSNR收敛
            ax1.plot(iterations, results[algo_name]['psnr_history'], 
                    label=algo_name, color=colors.get(algo_name, 'black'), 
                    linewidth=2)
            
            # SSIM收敛
            ax2.plot(iterations, results[algo_name]['ssim_history'], 
                    label=algo_name, color=colors.get(algo_name, 'black'), 
                    linewidth=2)
    
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR收敛曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM收敛曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppt3_convergence_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_all_images(original, corrupted, results):
    """保存所有图像"""
    # 创建输出目录
    output_dir = 'inpainting_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存原始图像
    cv2.imwrite(f'{output_dir}/original.png', (original * 255).astype(np.uint8))
    
    # 保存损坏图像
    cv2.imwrite(f'{output_dir}/corrupted.png', (corrupted * 255).astype(np.uint8))
    
    # 保存修复结果
    for algo_name, result in results.items():
        filename = f'{output_dir}/{algo_name.lower().replace("-", "_")}_restored.png'
        cv2.imwrite(filename, (result['restored'] * 255).astype(np.uint8))
        print(f"   已保存: {filename}")
    
    # 保存指标
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write("图像修复实验结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"原始图像尺寸: {original.shape}\n")
        f.write(f"损坏比例: 25%\n\n")
        
        for algo_name in ['ISTA', 'FISTA-L1', 'FISTA-TV', 'BM3D']:
            if algo_name in results:
                result = results[algo_name]
                psnr_display = "∞" if result['psnr'] == float('inf') else f"{result['psnr']:.2f}"
                f.write(f"{algo_name}:\n")
                f.write(f"  PSNR: {psnr_display} dB\n")
                f.write(f"  SSIM: {result['ssim']:.4f}\n")
                f.write(f"  时间: {result['time']:.2f}秒\n\n")

# 运行实验
if __name__ == "__main__":
    original, corrupted, results = run_complete_experiment()
    
    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
    print("\n生成的输出文件:")
    print("1. ppt3_inpainting_results.png - 结果对比图")
    print("2. ppt3_convergence_curves.png - 收敛曲线图")
    print("3. inpainting_results/目录 - 所有输出图像和指标")
    print("\nLaTeX表格已生成，可直接复制到报告中。")
