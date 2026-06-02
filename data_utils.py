import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Log1pTransform:
    """代替 Lambda(lambda x: torch.log1p(x))，支持多进程"""
    def __call__(self, x):
        return torch.log1p(x)
    
    def __repr__(self):
        return "Log1pTransform()"


def split_dataset(data_dir, test_size=0.1, val_size=0.1):
    print(">>> [Step 1] Scanning directory for CSV files...")
    csv_files = glob.glob(os.path.join(data_dir, "Zernike*.csv"))
    indices = [int(os.path.basename(f).replace("Zernike", "").replace(".csv", "")) for f in csv_files]
    print(f"    Found {len(indices)} samples in total.")

    train_idx, temp_idx = train_test_split(indices, test_size=(test_size + val_size), random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    print(f"    Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx

def split_train_val(data_dir, val_size=0.1):
    """ 仅划分训练集和验证集，不预留测试集（适用于有独立测试集文件夹的情况）"""
    print(">>>[Step 1] Scanning directory for CSV files (Train/Val only)...")
    indices = get_indices_from_dir(data_dir)
    print(f" Found {len(indices)} samples in total.")
    train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=42)
    print(f" Split sizes: Train={len(train_idx)}, Val={len(val_idx)} (Test set is external)")
    return train_idx, val_idx


def get_indices_from_dir(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "Zernike*.csv"))
    indices = [int(os.path.basename(f).replace("Zernike", "").replace(".csv", "")) for f in csv_files]
    return indices


def load_zernike_coeffs(filepath, num_modes=35):
    df = pd.read_csv(filepath, header=None)
    coeffs = df.values.flatten()[:num_modes]
    return coeffs



class ZernikeDataset(Dataset):
    def __init__(self, data_dir, indices, prefixes=["imgIF"], num_modes=35, use_log_preprocess=True):
        self.data_dir = data_dir
        self.indices = indices
        self.prefixes = prefixes
        self.num_modes = num_modes
        self.use_log_preprocess = use_log_preprocess

        transform_list = [
            transforms.ToTensor(),                     # PIL/np → [0,1] float tensor
        ]
        if self.use_log_preprocess:
            transform_list.append(Log1pTransform())
        

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        imgs = []
        for prefix in self.prefixes:
            img_path = os.path.join(self.data_dir, f"{prefix}{file_idx}.jpg")
            if not os.path.exists(img_path) and prefix.lower() == "imgnedf":
                img_path = os.path.join(self.data_dir, f"imgNedf{file_idx}.jpg")

            img = Image.open(img_path).convert('L')
            imgs.append(np.array(img))

        stacked = np.stack(imgs, axis=-1)          # (H, W, C)
        img_tensor = self.transform(stacked)       # 应用 log + Resize

        coeff_path = os.path.join(self.data_dir, f"Zernike{file_idx}.csv")
        coeffs = load_zernike_coeffs(coeff_path, self.num_modes)
        return img_tensor, torch.FloatTensor(coeffs)


class ZernikeDatasetFixed3Channel(Dataset):
    def __init__(self, data_dir, indices, input_types=["imgIF"], num_modes=35, use_log_preprocess=True):
        self.data_dir = data_dir
        self.indices = indices
        self.input_types = input_types
        self.num_modes = num_modes
        self.use_log_preprocess = use_log_preprocess

        transform_list = [
            transforms.ToTensor(),
        ]
        if self.use_log_preprocess:
            transform_list.append(Log1pTransform())
        

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        imgs = []

        # 通道 1: imgIF
        img_if_path = os.path.join(self.data_dir, f"imgIF{file_idx}.jpg")
        img_if = np.array(Image.open(img_if_path).convert('L'))
        imgs.append(img_if)

        # 通道 2: imgPoDF 或 全0
        if "imgPoDF" in self.input_types:
            img_podf_path = os.path.join(self.data_dir, f"imgPoDF{file_idx}.jpg")
            img_podf = np.array(Image.open(img_podf_path).convert('L'))
        else:
            img_podf = np.zeros_like(img_if)
        imgs.append(img_podf)

        # 通道 3: imgNeDF 或 全0
        if "imgNeDF" in self.input_types or "imgNedf" in self.input_types:
            img_nedf_path = os.path.join(self.data_dir, f"imgNeDF{file_idx}.jpg")
            if not os.path.exists(img_nedf_path):
                img_nedf_path = os.path.join(self.data_dir, f"imgNedf{file_idx}.jpg")
            img_nedf = np.array(Image.open(img_nedf_path).convert('L'))
        else:
            img_nedf = np.zeros_like(img_if)
        imgs.append(img_nedf)

        stacked = np.stack(imgs, axis=-1)
        img_tensor = self.transform(stacked)          # 应用 log + Resize

        coeff_path = os.path.join(self.data_dir, f"Zernike{file_idx}.csv")
        coeffs = load_zernike_coeffs(coeff_path, self.num_modes)
        return img_tensor, torch.FloatTensor(coeffs)


def visualize_sample(dataset, idx=0):
    img, coeff = dataset[idx]
    num_channels = img.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4))

    if num_channels == 1:
        axes = [axes]

    for i in range(num_channels):
        axes[i].imshow(img[i].numpy(), cmap='gray')
        if i < len(dataset.prefixes) if hasattr(dataset, 'prefixes') else True:
            axes[i].set_title(f"Channel {i} (log preprocessed)" if dataset.use_log_preprocess else f"Channel {i}")
    plt.tight_layout()
    plt.show()
