import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 无服务器环境下安全画图
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import subprocess
import math

# ========================================================
# 1. 基础神经网络组件 (含 CBAM & DoubleConv)
# ========================================================
class DoubleConv(nn.Module):
    """ 双层卷积块 (Conv -> BatchNorm -> ReLU) """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class CBAM(nn.Module):
    """ 经典通道+空间注意力机制 """
    def __init__(self, channels, ratio=16):
        super().__init__()
        r = max(1, channels // ratio)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        x = x * self.ca(x)
        # 空间注意力 (Concat 均值和最大值)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = self.sa(torch.cat([avg_out, max_out], dim=1))
        return x * spatial


# ========================================================
# 2. ConvLSTM 核心组件
# ========================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias
        )

    def forward(self, x_t, h_t, c_t):
        combined = torch.cat([x_t, h_t], dim=1)
        # 极简：使用 chunk 将 4 倍通道切分为 4 个门控信号
        cc_i, cc_f, cc_o, cc_g = torch.chunk(self.conv(combined), 4, dim=1)
        
        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)
        c_next = f * c_t + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        H, W = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1, bias=True):
        super().__init__()
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size, bias)
            for i in range(num_layers)
        ])

    def forward(self, x, hidden_state=None):
        B, T, _, H, W = x.shape
        device = x.device

        if hidden_state is None:
            hidden_state = [cell.init_hidden(B, (H, W), device) for cell in self.cell_list]

        cur_layer_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(T):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t], h, c)
                output_inner.append(h)
            cur_layer_input = torch.stack(output_inner, dim=1)

        return cur_layer_input, (h, c)


# ========================================================
# 3. 编解码网络子模块
# ========================================================
class EncoderBlock(nn.Module):
    """ 包含 2D卷积 -> ConvLSTM -> CBAM -> 池化 的时序编码器 """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.lstm = ConvLSTM(out_ch, out_ch)
        self.cbam = CBAM(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # 融合时序维度应用 2D 卷积
        x_flat = self.conv(x.view(B * T, C, H, W))
        C_out, H_out, W_out = x_flat.shape[1:]
        
        # 恢复时序输入 ConvLSTM
        x_seq, _ = self.lstm(x_flat.view(B, T, C_out, H_out, W_out))
        
        # 应用 CBAM 注意力机制与池化下采样
        x_flat = self.cbam(x_seq.view(B * T, C_out, H_out, W_out))
        x_pool_flat = self.pool(x_flat)
        H_p, W_p = x_pool_flat.shape[2:]

        return (
            x_flat.view(B, T, C_out, H_out, W_out),
            x_pool_flat.view(B, T, C_out, H_p, W_p)
        )


class BottleneckBlock(nn.Module):
    """ 瓶颈层 """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.lstm = ConvLSTM(out_ch, out_ch)
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = self.conv(x.view(B * T, C, H, W))
        C_out, H_out, W_out = x_flat.shape[1:]
        x_seq, _ = self.lstm(x_flat.view(B, T, C_out, H_out, W_out))
        x_flat = self.cbam(x_seq.view(B * T, C_out, H_out, W_out))
        return x_flat.view(B, T, C_out, H_out, W_out)


class DecoderBlock(nn.Module):
    """ 带跳跃连接的时空转置解码块 """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ========================================================
# 4. PSF 时序预测主模型
# ========================================================
class PSFConvLSTMPredictor(nn.Module):
    def __init__(self, seq_len=5, in_channels=2, base_channels=64, img_size=128):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        
        self.bottleneck = BottleneckBlock(base_channels * 4, base_channels * 8)
        
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        print(f"    ✅ PSFConvLSTMPredictor 初始化完成 (代码重构极简版)")

    def forward(self, x):
        # 编码器提取多尺度特征
        x1, x1_p = self.enc1(x)
        x2, x2_p = self.enc2(x1_p)
        x3, x3_p = self.enc3(x2_p)
        
        # 瓶颈层特征提炼
        x_b = self.bottleneck(x3_p)
        
        # 提取最后一个时间步特征用于解码器跳连 (UNet 机制)
        d3 = self.dec3(x_b[:, -1], x3[:, -1])
        d2 = self.dec2(d3, x2[:, -1])
        d1 = self.dec1(d2, x1[:, -1])
        
        return self.out_conv(d1)


# ========================================================
# 5. 数据集加载与指标计算
# ========================================================
class PSFSequenceDataset(Dataset):
    def __init__(self, data_dir, seq_len=5, img_size=128, start_idx=1, end_idx=None, use_log1p=True):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.img_size = img_size
        self.use_log1p = use_log1p
        self.start_idx = start_idx
        self.max_idx = end_idx if end_idx is not None else self._get_max_frame_idx()
        self.total_samples = self.max_idx - self.start_idx - self.seq_len + 1

        if self.total_samples <= 0:
            raise ValueError(f"数据帧不足：最大编号={self.max_idx}，序列长度={seq_len}")

    def _get_max_frame_idx(self):
        nums = []
        for f in os.listdir(self.data_dir):
            if f.startswith("imgIF") and f.endswith(".jpg"):
                try: nums.append(int(f.replace("imgIF", "").replace(".jpg", "")))
                except: pass
        return max(nums) if nums else 0

    def __len__(self):
        return self.total_samples

    def _load_img(self, idx, prefix="imgIF"):
        path = os.path.join(self.data_dir, f"{prefix}{idx}.jpg")
        img = Image.open(path).convert("L").resize((self.img_size, self.img_size), RESAMPLE_MODE)
        img = np.asarray(img, dtype=np.float32) / 255.0
        return np.log1p(img) if self.use_log1p else img

    def __getitem__(self, idx):
        seq_start = self.start_idx + idx
        input_frames = []
        for t in range(self.seq_len):
            frame_idx = seq_start + t
            img_if = self._load_img(frame_idx, "imgIF")
            img_podf = self._load_img(frame_idx, "imgPoDF")
            input_frames.append(np.stack([img_if, img_podf], axis=0))
        
        input_seq = np.stack(input_frames, axis=0)
        target = np.expand_dims(self._load_img(seq_start + self.seq_len, "imgIF"), axis=0)
        return torch.from_numpy(input_seq).float(), torch.from_numpy(target).float()


def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 100.0 if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def calc_ssim(pred, target):
    C1, C2 = 0.01**2, 0.03**2
    mu_p, mu_t = pred.mean(), target.mean()
    var_p, var_t = pred.var(), target.var()
    cov = ((pred - mu_p) * (target - mu_t)).mean()
    return ((2*mu_p*mu_t + C1) * (2*cov + C2)) / ((mu_p**2 + mu_t**2 + C1) * (var_p + var_t + C2)).item()


# ========================================================
# 6. 强化可视化工具
# ========================================================
class MetricTracker:
    """ 训练指标曲线跟踪与自动画图器 """
    def __init__(self):
        self.history = {"train_loss": [], "train_psnr": [], "val_loss": [], "val_psnr": [], "val_ssim": []}

    def update(self, t_loss, t_psnr, v_loss, v_psnr, v_ssim):
        # 自动识别并使用 .item() 剥离 Tensor，确保存入列表的全部是 float
        for key, val in zip(self.history.keys(), [t_loss, t_psnr, v_loss, v_psnr, v_ssim]):
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().item()
            self.history[key].append(val)

    def plot_curves(self, save_path):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(epochs, self.history["train_loss"], label="Train Loss", color="RoyalBlue")
        axes[0].plot(epochs, self.history["val_loss"], label="Val Loss", color="Tomato", linestyle="--")
        axes[0].set_title("Loss Convergence Curve")
        axes[0].set_xlabel("Epochs"), axes[0].set_ylabel("Loss")
        axes[0].grid(True), axes[0].legend()

        # PSNR
        axes[1].plot(epochs, self.history["train_psnr"], label="Train PSNR", color="RoyalBlue")
        axes[1].plot(epochs, self.history["val_psnr"], label="Val PSNR", color="Tomato", linestyle="--")
        axes[1].set_title("PSNR History (dB)")
        axes[1].set_xlabel("Epochs"), axes[1].set_ylabel("PSNR")
        axes[1].grid(True), axes[1].legend()

        # SSIM
        axes[2].plot(epochs, self.history["val_ssim"], label="Val SSIM", color="ForestGreen")
        axes[2].set_title("Validation SSIM Evolution")
        axes[2].set_xlabel("Epochs"), axes[2].set_ylabel("SSIM")
        axes[2].grid(True), axes[2].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


def compute_lcm(x, y):
    """ 计算最小公倍数，用于兼容较老 Python 版本的 GridSpec 布局计算 """
    greater = max(x, y)
    while True:
        if (greater % x == 0) and (greater % y == 0):
            return greater
        greater += 1



def plot_diagnostic_panel(inputs, target, pred, epoch, save_dir):
    """ 时序画廊 + 真值对比 + 绝对残差热力图 """
    os.makedirs(save_dir, exist_ok=True)
    seq_len = inputs.shape[0]
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, seq_len)

    # 1. 在焦历史帧
    for t in range(seq_len):
        ax = fig.add_subplot(gs[0, t])
        ax.imshow(inputs[t, 0].cpu().numpy(), cmap='inferno')
        ax.set_title(f"Input IF (t-{seq_len-1-t})")
        ax.axis('off')

    # 2. 离焦历史帧
    for t in range(seq_len):
        ax = fig.add_subplot(gs[1, t])
        ax.imshow(inputs[t, 1].cpu().numpy(), cmap='inferno')
        ax.set_title(f"Input PoDF (t-{seq_len-1-t})")
        ax.axis('off')

    # 3. 三合一诊断图
    col_w = max(1, seq_len // 3)
    
    # 真值
    ax_gt = fig.add_subplot(gs[2, 0:col_w])
    im_gt = ax_gt.imshow(target[0].cpu().numpy(), cmap='inferno')
    ax_gt.set_title("Ground Truth (t+1)")
    ax_gt.axis('off')
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

    # 预测
    ax_pd = fig.add_subplot(gs[2, col_w:2*col_w])
    im_pd = ax_pd.imshow(pred[0].cpu().numpy(), cmap='inferno')
    psnr_v = calc_psnr(pred.unsqueeze(0), target.unsqueeze(0))
    ssim_v = calc_ssim(pred.unsqueeze(0), target.unsqueeze(0))
    ax_pd.set_title(f"Prediction (t+1)\nPSNR: {psnr_v:.2f}dB | SSIM: {ssim_v:.4f}")
    ax_pd.axis('off')
    plt.colorbar(im_pd, ax=ax_pd, fraction=0.046, pad=0.04)

    # 绝对残差分布
    ax_er = fig.add_subplot(gs[2, 2*col_w:])
    err_map = torch.abs(pred[0] - target[0]).cpu().numpy()
    im_er = ax_er.imshow(err_map, cmap='coolwarm')
    ax_er.set_title("Absolute Error Map")
    ax_er.axis('off')
    plt.colorbar(im_er, ax=ax_er, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_diagnostic.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ========================================================
# 7. 训练与验证环
# ========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_psnr = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_psnr += calc_psnr(pred, y)
    return total_loss / len(loader), total_psnr / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch=0, save_dir=None):
    model.eval()
    total_loss, total_psnr, total_ssim = 0.0, 0.0, 0.0
    save_first = True

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        total_loss += loss.item()
        total_psnr += calc_psnr(pred, y)
        total_ssim += calc_ssim(pred, y)

        if save_first and save_dir is not None:
            plot_diagnostic_panel(x[0], y[0], pred[0], epoch, save_dir)
            save_first = False

    return total_loss / len(loader), total_psnr / len(loader), total_ssim / len(loader)


# ========================================================
# 8. 主流程
# ========================================================
def main():
    cfg = {
        "data_dir": "./dataset/ao_simulated",
        "seq_len": 5,
        "img_size": 128,
        "base_channels": 64,
        "batch_size": 8,
        "num_epochs": 50,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_log1p": True,
        "train_ratio": 0.8,
        "weight_save_dir": "./weights/psf_prediction",
        "vis_save_dir": "./results/psf_prediction"
    }

    os.makedirs(cfg["weight_save_dir"], exist_ok=True)
    os.makedirs(cfg["vis_save_dir"], exist_ok=True)

    print(">>> 加载数据集...")
    temp_ds = PSFSequenceDataset(cfg["data_dir"], cfg["seq_len"], cfg["img_size"], use_log1p=cfg["use_log1p"])
    train_num = int(len(temp_ds) * cfg["train_ratio"])

    train_ds = PSFSequenceDataset(
        cfg["data_dir"], cfg["seq_len"], cfg["img_size"],
        start_idx=1, end_idx=1 + train_num + cfg["seq_len"] - 1, use_log1p=cfg["use_log1p"]
    )
    val_ds = PSFSequenceDataset(
        cfg["data_dir"], cfg["seq_len"], cfg["img_size"],
        start_idx=1 + train_num, end_idx=temp_ds.max_idx, use_log1p=cfg["use_log1p"]
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)
    print(f"    训练集: {len(train_ds)} 样本 | 验证集: {len(val_ds)} 样本")

    print(">>> 初始化网络结构...")
    model = PSFConvLSTMPredictor(
        seq_len=cfg["seq_len"], in_channels=2, base_channels=cfg["base_channels"], img_size=cfg["img_size"]
    ).to(cfg["device"])

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])
    tracker = MetricTracker()

    best_psnr = 0.0
    print(f"\n>>> 开始训练，共 {cfg['num_epochs']} 轮...")
    for epoch in range(1, cfg["num_epochs"] + 1):
        train_loss, train_psnr = train_one_epoch(model, train_loader, criterion, optimizer, cfg["device"])
        val_loss, val_psnr, val_ssim = evaluate(model, val_loader, criterion, cfg["device"], epoch, cfg["vis_save_dir"])
        scheduler.step()

        # 记录指标并绘制曲线
        tracker.update(train_loss, train_psnr, val_loss, val_psnr, val_ssim)
        tracker.plot_curves(os.path.join(cfg["vis_save_dir"], "training_metrics.png"))

        print(f"Epoch {epoch}/{cfg['num_epochs']} | "
              f"Train Loss: {train_loss:.6f} (PSNR: {train_psnr:.2f}dB) | "
              f"Val Loss: {val_loss:.6f} (PSNR: {val_psnr:.2f}dB, SSIM: {val_ssim:.4f})")

        # 保存最优模型
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(cfg["weight_save_dir"], "model_best.pth"))
            print(f"    ⭐ 保存最优模型，PSNR 达到: {best_psnr:.2f}dB")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(cfg["weight_save_dir"], f"epoch_{epoch}.pth"))

    print("\n>>> 训练成功完成！")


if __name__ == "__main__":
    main()
