import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
from sklearn.model_selection import train_test_split

# 从您的训练脚本中导入模型和数据集类
from train_shwfs import ZernikeNet, SHWFSDataset

plt.style.use('seaborn-v0_8-paper')
sns.set_context("talk")


def split_dataset(data_dir, test_size=0.1, val_size=0.1):
    """
    扫描目录并划分数据集索引 (固定随机种子 random_state=42 保证每次划分一致)
    """
    print(">>> [Step 1] 扫描目录中的 CSV 文件...")
    csv_files = glob.glob(os.path.join(data_dir, "Zernike*.csv"))[cite: 3]
    
    # 提取文件编号，兼容 Zernike0001.csv 这种补零格式
    indices = [int(os.path.basename(f).replace("Zernike", "").replace(".csv", "")) for f in csv_files][cite: 3]
    print(f"    共找到 {len(indices)} 个样本.")

    # 第一次划分：分出训练集和 (验证+测试) 临时集
    train_idx, temp_idx = train_test_split(indices, test_size=(test_size + val_size), random_state=42)[cite: 3]
    
    # 第二次划分：将临时集平分为验证集和测试集
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)[cite: 3]
    
    print(f"    数据集划分完成: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx[cite: 3]


def test_and_plot_shwfs():
    # ==========================================
    # --- 1. 参数配置 ---
    # ==========================================
    test_dir = "./dataset/shwfs_data"  # 测试集数据路径
    num_modes = 15                     # Zernike 阶数
    batch_size = 16
    num_visualize = 10                 # 设定要单独绘制对比图和保存SHWFS图的样本数量
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = './results_shwfs'
    samples_dir = os.path.join(results_dir, 'samples_plots')
    os.makedirs(samples_dir, exist_ok=True)

    # SHWFS 图像预处理 (与 train_shwfs.py 保持一致)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print(f">>> 正在从 {test_dir} 加载测试数据...")
    _, _, test_idx = split_dataset(test_dir, test_size=0.1, val_size=0.1)
    test_dataset = SHWFSDataset(data_dir=test_dir, indices=test_idx, num_zernike=num_modes, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- 2. 加载模型 ---
    model = ZernikeNet(num_outputs=num_modes, in_channels=1).to(device)
    model_weight_path = "./weights/model_best.pth"
    
    if os.path.exists(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print(f">>> 成功加载权重: {model_weight_path}")
    else:
        print(f">>> [警告] 未找到权重文件: {model_weight_path}，将使用随机初始化权重进行测试。")
        
    model.eval()

    # --- 3. 执行推理 (含测速和图像收集) ---
    all_preds, all_trues, all_latencies = [], [], []
    all_imgs = [] 
    
    print(f">>> 开始对 {len(test_loader)} 个样本进行推理分析 (设备: {device})...")

    # GPU 预热
    if device.type == 'cuda':
        dummy_input = torch.randn(1, 1, 480, 480).to(device)
        with torch.no_grad():
            for _ in range(3):
                model(dummy_input)
        torch.cuda.synchronize()

    with torch.no_grad():
        for imgs, coeffs in tqdm(test_loader):
            imgs_gpu = imgs.to(device)
            batch_size_current = imgs.size(0)

            # 计时开始
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.perf_counter()

            outputs = model(imgs_gpu)

            # 计时结束
            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.perf_counter()

            # 计算单样本平均时延并记录
            per_sample_time = (end_time - start_time) / batch_size_current
            all_latencies.extend([per_sample_time] * batch_size_current)

            preds = outputs.cpu().numpy()
            trues = coeffs.numpy()
            
            all_preds.append(preds)
            all_trues.append(trues)
            
            # 收集图像用于后续可视化 (只收集前 num_visualize 个批次以节省内存)
            if len(all_imgs) * batch_size < num_visualize:
                all_imgs.append(imgs.numpy())

    pred_np = np.concatenate(all_preds, axis=0)
    true_np = np.concatenate(all_trues, axis=0)
    imgs_np = np.concatenate(all_imgs, axis=0) if len(all_imgs) > 0 else np.array([])

    # 符号一致性计算
    sign_match = np.sign(true_np) * np.sign(pred_np)
    mismatch = sign_match < 0
    ratio_item_err = np.mean(mismatch)

    wrong_mag = np.abs(pred_np[mismatch])
    avg_wrong_mag = np.mean(wrong_mag) if len(wrong_mag) > 0 else 0.0

    threshold = 0.01
    severe_mask = mismatch & (np.abs(pred_np) > threshold)
    severe_ratio = np.mean(severe_mask)

    mean_sign_prod = np.mean(pred_np * true_np)
    norm_sign_err = np.sum(np.abs(pred_np - true_np)[mismatch]) / (np.sum(np.abs(true_np)) + 1e-8)

    # --- 4. 核心计算、数据保存与汇总报告 ---
    sample_data_list = []
    sample_mse_list = []
    sample_r2_list = []

    # 逐样本计算各项指标后再进行汇总
    for i in range(len(true_np)):
        sample_id = i + 1  # 对应文件名中的索引
        
        # 单独计算每个样本的 MSE 和 R2
        s_mse = mean_squared_error(true_np[i], pred_np[i])
        s_r2 = r2_score(true_np[i], pred_np[i])
        
        sample_mse_list.append(s_mse)
        sample_r2_list.append(s_r2)
        
        s_sign_errors = np.sum(mismatch[i])
        s_latency = all_latencies[i]

        row = {'Sample_ID': sample_id}
        for m in range(num_modes): row[f'Pred_Z{m}'] = pred_np[i][m]
        for m in range(num_modes): row[f'True_Z{m}'] = true_np[i][m]
        row['Sample_MSE'] = s_mse
        row['Sample_R2'] = s_r2
        row['Sign_Error_Count'] = s_sign_errors
        row['Avg_Wrong_Mag'] = np.mean(np.abs(pred_np[i][mismatch[i]])) if np.any(mismatch[i]) else 0.0
        row['Severe_Sign_Err'] = np.mean(severe_mask[i])
        row['Inference_Latency_sec'] = s_latency
        sample_data_list.append(row)

    df_samples = pd.DataFrame(sample_data_list)
    df_samples.to_csv(os.path.join(results_dir, 'test_samples_results.csv'), index=False)

    # 针对逐样本指标求平均值
    avg_sample_mse = np.mean(sample_mse_list)
    avg_sample_r2 = np.mean(sample_r2_list)
    avg_latency_ms = np.mean(all_latencies) * 1000

    # 生成并保存测试总结报告
    summary_text = (
        "================ 评估报告 ================\n"
        f"测试样本总数: {len(true_np)}\n"
        f"平均 MSE (逐样本计算后求均值): {avg_sample_mse:.6f}\n"
        f"平均 R2 (逐样本计算后求均值) : {avg_sample_r2:.6f}\n"
        f"符号错误比例: {ratio_item_err:.2%}\n"
        f"Avg Wrong Magnitude: {avg_wrong_mag:.4f}\n"
        f"严重符号错误 (>0.01): {severe_ratio:.2%}\n"
        f"Mean Sign Product: {mean_sign_prod:.4f}\n"
        f"Norm Sign Error Contribution: {norm_sign_err:.2%}\n"
        f"平均单样本推理时延: {avg_latency_ms:.2f} ms\n"
        "==================================================\n"
    )

    print(f"\n{summary_text}")
    with open(os.path.join(results_dir, 'test_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)

    # ==========================================
    # --- 5. 绘图：绝对 RMSE、散点图与逐样本对比 ---
    # ==========================================
    print(">>> 开始生成可视化图表...")

    # 图表 1: 绝对 RMSE
    rmse_per_mode = np.sqrt(np.mean((true_np - pred_np) ** 2, axis=0))
    plt.figure(figsize=(10, 5))
    modes = [f"Z{i}" for i in range(num_modes)]
    plt.bar(modes, rmse_per_mode, color=sns.color_palette("viridis", len(modes)), alpha=0.8)
    plt.title("Absolute RMSE per Zernike Mode", fontsize=14)
    plt.ylabel("RMSE (Lower is better)")
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "analysis_rmse_error.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    # 图表 2: 全局真实值 vs 预测值散点图 (包含精确象限颜色蒙层)
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 9))

    all_vals = np.concatenate([true_np.flatten(), pred_np.flatten()])
    min_val = all_vals.min() * 1.1 
    max_val = all_vals.max() * 1.1

    # 使用浅绿色蒙层高亮同号(正确)区域
    correct_sign_color = '#d4f1f4'

    # 填充第一象限 (X>0, Y>0)
    ax.fill_between([0, max_val], 0, max_val, color=correct_sign_color, alpha=0.5)
    # 填充第三象限 (X<0, Y<0)
    ax.fill_between([min_val, 0], min_val, 0, color=correct_sign_color, alpha=0.5)

    # 绘制散点
    ax.scatter(true_np.flatten(), pred_np.flatten(), alpha=0.2, s=10, color='#1f77b4', zorder=2)
    # 绘制对角拟合线
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit (y=x)", zorder=3)

    # 辅助线和标注
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, zorder=1) 
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5, zorder=1) 
    ax.text(max_val * 0.95, max_val * 0.95, "Correct Sign", color='#16a085', fontsize=12, ha='right', va='top')
    ax.text(min_val * 0.95, min_val * 0.95, "Correct Sign", color='#16a085', fontsize=12, ha='left', va='bottom')

    ax.set_title("Global True vs Predicted Zernike Coefficients")
    ax.set_xlabel("True Coefficients")
    ax.set_ylabel("Predicted Coefficients")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.3, zorder=1)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "analysis_scatter_global.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    # 图表 3: 逐样本柱状图对比 & SHWFS 图像保存
    # --------------------------------------------------
    actual_visualize = min(num_visualize, len(true_np))
    print(f">>> 正在为前 {actual_visualize} 个样本保存对比图和 SHWFS 输入图...")
    
    for i in range(actual_visualize):
        sample_id = i + 1

        # 3.1 保存系数对比柱状图
        plt.figure(figsize=(12, 5))
        x_axis = np.arange(num_modes)
        width = 0.35
        plt.bar(x_axis - width / 2, true_np[i], width, label='True Values', alpha=0.8, color='#2ca02c')
        plt.bar(x_axis + width / 2, pred_np[i], width, label='Predicted Values', alpha=0.8, color='#d62728')
        plt.title(f"Sample {sample_id} Zernike Coefficients Comparison")
        plt.xlabel("Zernike Mode Index")
        plt.ylabel("Coefficient Value")
        plt.xticks(x_axis, modes)
        plt.legend()
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f"sample_{sample_id}_coeffs.png"), dpi=300)
        plt.close()

        # 3.2 保存对应的单通道 SHWFS 图像
        if len(imgs_np) > i:
            # 维度转换: (C, H, W) -> (H, W, C)
            img_data = imgs_np[i].transpose(1, 2, 0)
            
            plt.figure(figsize=(6, 6))
            # SHWFS 为单通道灰度图，取第 0 通道绘制
            plt.imshow(img_data[:, :, 0], cmap='gray')
            plt.title(f"SHWFS Input - Sample {sample_id}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f"sample_{sample_id}_shwfs.png"), dpi=300, bbox_inches='tight')
            plt.close()

    print(f">>> [已完成] 所有分析结果、TXT总结及图表已保存至 {results_dir}")

if __name__ == "__main__":
    test_and_plot_shwfs()
