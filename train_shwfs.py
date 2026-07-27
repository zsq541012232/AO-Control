import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import matplotlib 
matplotlib.use('Agg')


class ConvBnReLU(nn.Module):
    """基础卷积块：Conv -> BN -> ReLU，兼容 TensorRT 部署"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SIRNet1_Block(nn.Module):
    """SIR-Net1 多尺度残差核心模块"""
    def __init__(self, in_channels, out_channels):
        super(SIRNet1_Block, self).__init__()
        inter_channels = out_channels // 4
        
        self.branch1 = ConvBnReLU(in_channels, inter_channels, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBnReLU(in_channels, inter_channels, kernel_size=1),
            ConvBnReLU(inter_channels, inter_channels, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvBnReLU(in_channels, inter_channels, kernel_size=1),
            ConvBnReLU(inter_channels, inter_channels, kernel_size=5, padding=2)
        )
        
        self.concat_conv = nn.Sequential(
            nn.Conv2d(inter_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.concat_conv(out)
        out = out + self.shortcut(x)
        return self.relu(out)

class ZernikeNet(nn.Module):
    """
    专为 SHWFS (14x14子孔径, 480x480分辨率) 设计的 SIR-Net
    """
    def __init__(self, num_outputs=15, in_channels=1, weight_path=None):
        super(ZernikeNet, self).__init__()
        
        # 1. 初始降采样 (Stem)
        # 输入: 480x480
        self.stem = nn.Sequential(
            ConvBnReLU(in_channels, 64, kernel_size=7, stride=2, padding=3), # -> 240x240
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                # -> 120x120
            ConvBnReLU(64, 128, kernel_size=3, stride=1, padding=1),
        )
        
        # 2. 特征提取与穿插空间降采样 (精准对齐 14x14 阵列)
        self.block1 = SIRNet1_Block(128, 256)
        self.pool1 = nn.MaxPool2d(2, 2) # 120x120 -> 60x60
        
        self.block2 = SIRNet1_Block(256, 256)
        self.pool2 = nn.MaxPool2d(2, 2) # 60x60 -> 30x30
        
        self.block3 = SIRNet1_Block(256, 512)
        self.pool3 = nn.MaxPool2d(2, 2) # 30x30 -> 15x15 (物理意义: 这里每个像素感受野近似对应1个子孔径!)
        
        self.block4 = SIRNet1_Block(512, 512) # 保持 15x15

        # 3. 通道压缩 (为全连接层减负，防止参数量过大)
        self.channel_compress = ConvBnReLU(512, 64, kernel_size=1) 
        
        # 4. SIR-Net2 多层感知机
        # 展平后维度: 15 * 15 * 64 = 14400
        self.sirnet2 = nn.Sequential(
            nn.Linear(14400, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_outputs)
        )
        
        if weight_path is not None:
            try:
                self.load_state_dict(torch.load(weight_path))
                print(f"成功加载预训练权重: {weight_path}")
            except Exception as e:
                print(f"未能加载权重: {e}，使用随机初始化。")

    def forward(self, x):
        # 提取并下采样
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        
        # 压缩通道并保留 15x15 空间结构进行展平
        x = self.channel_compress(x)
        x = torch.flatten(x, 1)
        
        # 映射到 Zernike 系数
        x = self.sirnet2(x)
        return x

# ==========================================
# 1. 自定义哈特曼数据集 Dataset 类
# ==========================================
class SHWFSDataset(Dataset):
    def __init__(self, data_dir, num_samples, num_zernike=15, transform=None):
        """
        初始化数据集
        :param data_dir: 图像和标签的文件夹路径
        :param num_samples: 数据集样本总数
        :param num_zernike: Zernike 模式数量 (默认15)
        :param transform: 图像预处理
        """
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.num_zernike = num_zernike
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 索引通常从 1 开始
        file_idx = idx + 1
        
        # 组装图片名格式: image0001.jpg, image0002.jpg ... (符合 image%04d.jpg)
        # 如果您的图片是 image   1.jpg (空格填充), 可以改为 f"image{file_idx:4d}.jpg"
        img_name = f"image{file_idx:04d}.jpg"
        img_path = os.path.join(self.data_dir, img_name)
        
        # 加载哈特曼灰度图
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)

        # 加载标签 (Zernike系数)
        # 假设标签文件也是独立的，比如 Zernike0001.csv
        label_name = f"Zernike{file_idx:04d}.csv"
        label_path = os.path.join(self.data_dir, label_name)
        
        if os.path.exists(label_path):
            # 从CSV读取系数，假设逗号分隔或换行
            label = np.loadtxt(label_path, delimiter=',')
            label = label[:self.num_zernike]
        else:
            # 兼容性处理：如果没有找到，返回全0（请根据实际数据格式修改）
            label = np.zeros(self.num_zernike, dtype=np.float32)

        label = torch.tensor(label, dtype=torch.float32)
        return image, label

# ==========================================
# 2. 训练验证主函数
# ==========================================
def train_and_test():
    # --- 超参数配置 ---
    DATA_DIR = "./dataset/shwfs_data"  # 替换为您的数据路径
    NUM_SAMPLES = 5000                 # 您拥有的图像总数
    NUM_ZERNIKE = 15                   # Zernike 阶数
    BATCH_SIZE = 16                    # 批次大小 (480x480 比较吃显存，建议设小一点)
    EPOCHS = 50                        # 训练轮次
    LR = 1e-4                          # 学习率
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确保保存权重的文件夹存在
    os.makedirs("./weights", exist_ok=True)

    # --- 图像预处理 ---
    # 由于是 480x480 的 SHWFS，不要进行中心裁剪，直接转张量并归一化，保留所有 14x14 的子孔径信息
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 简单的归一化，视情况可改为 transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # --- 数据加载与划分 ---
    print(">>> 正在加载数据集...")
    full_dataset = SHWFSDataset(DATA_DIR, num_samples=NUM_SAMPLES, num_zernike=NUM_ZERNIKE, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 初始化模型与优化器 ---
    model = ZernikeNet(num_outputs=NUM_ZERNIKE, in_channels=1).to(DEVICE)
    criterion = nn.MSELoss() # 波前复原属于回归任务，使用均方误差
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # 学习率调度器：如果验证集 loss 不下降，降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # --- 训练循环 ---
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    print(f">>> 开始训练 (设备: {DEVICE})")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_loss / train_size
        train_losses.append(epoch_train_loss)

        # --- 验证循环 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
        epoch_val_loss = val_loss / val_size
        val_losses.append(epoch_val_loss)
        
        # 调整学习率
        scheduler.step(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "./weights/model_best.pth")
            print(f"  --> 模型已保存 (Val Loss 改善至 {best_val_loss:.6f})")

    # --- 测试与绘图 ---
    print("\n>>> 训练完成！开始绘制 Loss 曲线...")
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('SIR-Net Training Curve (SHWFS)')
    plt.legend()
    plt.grid(True)
    plt.savefig("./training_curve.png", dpi=150)
    print(">>> 曲线已保存为 training_curve.png")

if __name__ == "__main__":
    train_and_test()
