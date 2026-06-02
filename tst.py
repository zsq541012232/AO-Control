from pyexpat import model

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import split_dataset, get_indices_from_dir, ZernikeDataset, ZernikeDatasetFixed3Channel
from model import (ConsistentUnderCorrectLoss, SignMarginShrinkLoss, ZernikeNet, ZernikeViT, ZernikeEffNet, SignWeightedMSELoss, SignMarginLoss,
                   ZernikeSiameseViTAttnResRoPE, ZernikeSiameseResNetCBAM, ZernikeDualCrossNet, ZernikeUNetMambaDeepFusion, ZernikeFusionMambaPure, 
                   ZernikeMambaPure, ZernikeUNet)
import pandas as pd
import os
from sklearn.metrics import r2_score, mean_squared_error
import time

# add
from torch_npu.contrib import transfer_to_npu
import matplotlib
matplotlib.use('Agg')

plt.style.use('seaborn-v0_8-paper')
sns.set_context("talk")


def test_and_plot():
    # ==========================================
    # --- 1. 参数配置 ---
    # ==========================================
    use_fixed_3channel = False  # 保持与 train.py 配置对应
    use_independent_test_dir = False

    test_dir = "dataset/ao_simulated"
    # test_dir = "../dataset/train/imgData-r6-35"

    num_modes = 15
    batch_size = 32
    # prefixes 定义了输入图像的类型和顺序
    # prefixes = ["imgIF", "imgPoDF", "imgNeDF"]  # Channel 0: In-Focus, Channel 1: Post-Defocus
    prefixes = ["imgIF", "imgPoDF"]  # Channel 0: In-Focus, Channel 1: Post-Defocus


    num_visualize = 10  # 设定要单独绘制对比图和保存PSF的样本数量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = './results'
    samples_dir = os.path.join(results_dir, 'samples_plots')
    os.makedirs(samples_dir, exist_ok=True)

    # 数据集准备
    if use_independent_test_dir:
        test_idx = get_indices_from_dir(test_dir)
        print(f">>> Independent Test Mode: Using all {len(test_idx)} samples from {test_dir}")
    else:
        _, _, test_idx = split_dataset(test_dir)
    data_dir_test = test_dir

    if use_fixed_3channel:
        # 使用 input_types 指明实际传入的图像
        test_dataset = ZernikeDatasetFixed3Channel(data_dir_test, test_idx, input_types=prefixes, num_modes=num_modes)
        model_in_channels = 3
    else:
        test_dataset = ZernikeDataset(data_dir=data_dir_test,
                                      indices=test_idx,
                                      prefixes=prefixes,
                                      num_modes=num_modes,
                                      use_log_preprocess=False)
        model_in_channels = len(prefixes)

    # 注意：为了后面方便提取单个样本的图像进行保存，这里将 shuffle 设为 False
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- 2. 加载模型 ---
    if use_fixed_3channel:
        # model = ZernikeViT(num_outputs=num_modes, in_channels=model_in_channels, weight_path=weight_path).to(device)   # vit
        # model = ZernikeNet(num_outputs=num_modes, in_channels=model_in_channels, weight_path=weight_path).to(device)   # resnet+cbam
        # model = ZernikeViTAttnResRoPE(num_outputs=num_modes, in_channels=model_in_channels).to(device)
        raise NotImplementedError("3通道模式暂未适配 Siamese 模型")
    else:
        # model = ZernikeSiameseViTAttnResRoPE(num_outputs=num_modes).to(device)
        # model = ZernikeEffNet(num_outputs=num_modes, in_channels=model_in_channels, weight_path=None).to(device)
        model = ZernikeNet(num_outputs=num_modes, in_channels=model_in_channels).to(device)
        # model = ZernikeSiameseResNetCBAM(num_outputs=num_modes).to(device)
        # model = ZernikeDualCrossNet(num_outputs=num_modes).to(device)
        # model = ZernikeUNet(num_outputs=num_modes, in_channels=model_in_channels).to(device)
        # model = ZernikeMambaPure(num_outputs=num_modes, in_channels=model_in_channels, img_size=128).to(device)
        # model = ZernikeFusionMambaPure(num_outputs=num_modes, in_channels=model_in_channels).to(device)
        # model = ZernikeUNetMambaDeepFusion(num_outputs=num_modes, in_channels=model_in_channels, img_size=128).to(device)
  
    model_name = model.__class__.__name__
    model_weight_path = "./weights/model_best.pth"
    if os.path.exists(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print(f">>> 成功加载权重: {model_weight_path}")
    model.eval()

    # --- 3. 执行推理 (含测速和图像收集) ---
    all_preds, all_trues, all_latencies = [], [], []
    all_imgs = []  # 新增：用于收集 raw 图像数据用于后续保存 PSF
    print(f">>> 开始对 {len(test_idx)} 个样本进行推理分析...")

    # GPU 预热
    if device.type == 'cuda':
        dummy_input = torch.randn(1, model_in_channels, 128, 128).to(device)
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
            all_imgs.append(imgs.numpy())  # 收集 CPU 端的图像数据

    pred_np = np.concatenate(all_preds, axis=0)
    true_np = np.concatenate(all_trues, axis=0)
    imgs_np = np.concatenate(all_imgs, axis=0)  # (N, C, H, W)

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

    for i in range(len(true_np)):
        sample_id = test_idx[i]
        s_mse = mean_squared_error(true_np[i], pred_np[i])
        s_r2 = r2_score(true_np[i], pred_np[i])
        s_sign_errors = np.sum(mismatch[i])
        s_latency = all_latencies[i]

        sample_mse_list.append(s_mse)
        sample_r2_list.append(s_r2)

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

    avg_sample_mse = np.mean(sample_mse_list)
    avg_sample_r2 = np.mean(sample_r2_list)
    avg_latency_ms = np.mean(all_latencies) * 1000

    # 生成并保存测试总结报告
    summary_text = (
        "================ 评估报告 ================\n"
        f"输入通道配置: {prefixes}\n"
        f"测试样本总数: {len(test_idx)}\n"
        f"平均 MSE: {avg_sample_mse:.6f}\n"
        f"平均 R2 : {avg_sample_r2:.6f}\n"
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
    # 图表 2: 全局真实值 vs 预测值散点图 (新增符号蒙层)
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 9))

    # 获取坐标轴范围以便铺设蒙层
    all_vals = np.concatenate([true_np.flatten(), pred_np.flatten()])
    min_val = all_vals.min() * 1.1  # 稍微留一点边缘
    max_val = all_vals.max() * 1.1

    # --- [核心修改 1] 绘制符号一致区域的透明蒙层 ---
    # 使用浅绿色 (#d4f1f4) 代表符号一致（Correct Sign）区域，透明度设为 0.5
    correct_sign_color = '#d4f1f4'

    # 第一象限: X>0, Y>0
    ax.fill_between([0,max_val],0, max_val, color=correct_sign_color, alpha=0.5)
    # 第三象限: X<0, Y<0
    ax.fill_between([min_val,0],min_val, 0, color=correct_sign_color, alpha=0.5)


    # 绘制散点，稍微调小点并增加透明度以应对数据量大的情况
    ax.scatter(true_np.flatten(), pred_np.flatten(), alpha=0.2, s=10, color='#1f77b4', zorder=1)

    # 绘制对角完美拟合线 y = x
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit (y=x)", zorder=2)

    # 辅助线和标注
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, zorder=1)  # X轴
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5, zorder=1)  # Y轴
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
    plt.savefig(os.path.join(results_dir, "analysis_scatter_global_with_sign_mask.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    # 图表 3: 逐样本柱状图对比 & PSF 图像保存 (恢复并修改)
    # --------------------------------------------------
    print(f">>> 正在为前 {min(num_visualize, len(test_idx))} 个样本保存对比图和 PSF 拼图...")
    for i in range(min(num_visualize, len(test_idx))):
        sample_id = test_idx[i]

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

        # --- 保存对应的 PSF 图像横向拼图 ---
        # 提取图像数据 (C, H, W) -> (H, W, C) 用于 matplotlib 显示
        # 如果是单通道，squeeze 掉 C 通道
        img_data = imgs_np[i].transpose(1, 2, 0)  # (224, 224, C)

        # --- 保存 PSF 拼图（支持 2 或 3 通道，动态子图）---
        num_c = img_data.shape[-1]
        fig_psf, axes_psf = plt.subplots(1, num_c, figsize=(6 * num_c, 4), squeeze=False)
        axes_psf = axes_psf.flatten()

        titles = ["In-Focus (imgIF)", "Post-Defocus (imgPoDF)", "Negative-Defocus (imgNeDF)"]
        for c in range(num_c):
            axes_psf[c].imshow(img_data[:, :, c], cmap='gray')
            axes_psf[c].set_title(titles[c])
            axes_psf[c].axis('off')

        plt.suptitle(f"PSF Images for Sample {sample_id}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(samples_dir, f"sample_{sample_id}_psf.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_psf)

    print(f">>> [已完成] 所有分析结果、TXT总结及图表（含PSF拼图）已保存至 {results_dir}")


if __name__ == "__main__":
    test_and_plot()
