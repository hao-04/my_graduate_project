"""
骨骼步态数据集  — 数据加载与预处理
=============================================
数据源:
    - data/normal_LsideSegm_28markers.pkl   正常人左侧步态 (138 人)
    - data/stroke_NsideSegm_28markers.pkl   中风患者健侧步态 (50 人)

原始格式:  每人 data[i].shape = (28, n_trials_i, 1001, 3)
           即 (Markers, Trials, Frames, XYZ)

目标格式:  N × 1001 × 28 × 3  (样本, 帧, 关节, 坐标)

坐标约定 (Vicon):
    dim 0 — X 前进方向
    dim 1 — Y 左右方向 (内外侧)
    dim 2 — Z 垂直方向 (高度)
    用户文档中 "y轴" → 实际 dim2 (Z),"xz平移" → 实际 dim0,dim1 (X,Y)
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit


# -------------------------------------------------------------------
# 关节索引常量
# -------------------------------------------------------------------
FOOT_INDICES = [14, 15, 26, 27]   # L_HEE, L_TOE, R_HEE, R_TOE
PELVIS_INDICES = [4, 16]          # L_ASI, R_ASI
VERTICAL_AXIS = 2                 # Z — 垂直方向
HORIZONTAL_AXES = [0, 1]          # X, Y — 水平面


def is_valid_raw_sample(sample: np.ndarray, eps: float = 1e-8) -> bool:
    """
    判断原始 trial 是否有效，过滤 NaN/Inf 与近似全 0 样本。

    sample: (F, M, D)
    """
    if not np.isfinite(sample).all():
        return False

    # 过滤近似全 0 样本，避免后续归一化退化
    if np.max(np.abs(sample)) <= eps:
        return False

    return True


def is_valid_processed_sample(sample: np.ndarray, eps: float = 1e-8) -> bool:
    """
    判断预处理后的样本是否有效，过滤 NaN/Inf 与垂直方向无变化样本。

    sample: (F, M, D)
    """
    if not np.isfinite(sample).all():
        return False

    # 过滤 Z 轴无变化（或几乎无变化）样本
    z_span = sample[:, :, VERTICAL_AXIS].max() - sample[:, :, VERTICAL_AXIS].min()
    if z_span <= eps:
        return False

    return True


# ===================================================================
#  预处理
# ===================================================================

def preprocess_sample(sample: np.ndarray) -> np.ndarray:
    """
    对单个样本进行空间预处理（就地修改副本）。

    sample: (F, M, D) = (1001, 28, 3)
    return: (1001, 28, 3) 预处理后
    """
    sample = sample.copy()
    F, M, D = sample.shape

    # ---- 1. 垂直平移 (Z 轴) ----
    # 取 4 个足部关键点在所有帧中的 Z 轴最小值，令其为 0
    foot_z = sample[:, FOOT_INDICES, VERTICAL_AXIS]   # (F, 4)
    z_min = foot_z.min()
    sample[:, :, VERTICAL_AXIS] -= z_min

    # ---- 2. 水平平移 (X, Y 轴) ----
    # 取左右 ASI 在所有帧中 X, Y 轴的平均值，令骨盆中心为原点
    pelvis_xy = sample[:, PELVIS_INDICES][:, :, HORIZONTAL_AXES]  # (F, 2, 2)
    xy_mean = pelvis_xy.mean(axis=(0, 1))                         # (2,)  [mean_x, mean_y]
    sample[:, :, HORIZONTAL_AXES] -= xy_mean

    # ---- 3. 身高归一化 ----
    # 所有关节 Z 轴值除以该样本所有帧中 Z 轴最大值
    z_max = sample[:, :, VERTICAL_AXIS].max()
    if z_max > 1e-6:
        sample[:, :, VERTICAL_AXIS] /= z_max

    return sample


def zscore_normalize(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score 归一化。

    data: (N, F, M, D)
    mean, std: 如果提供则使用（用于验证/测试集），否则从 data 计算。
    return: (归一化数据, mean, std)
    """
    if mean is None:
        mean = data.mean(axis=(0, 1, 2), keepdims=True)  # (1, 1, 1, D)
    if std is None:
        std = data.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    return (data - mean) / std, mean.squeeze(), std.squeeze()


# ===================================================================
#  数据加载
# ===================================================================

def _load_pkl(filepath: str) -> dict:
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_and_prepare(
    normal_path: str = "data/normal_LsideSegm_28markers.pkl",
    stroke_path: str = "data/stroke_NsideSegm_28markers.pkl",
    apply_zscore: bool = False,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """
    加载两个 pkl 文件，整理为 N×F×M×D 并按受试者级别划分训练/验证集。

    Parameters
    ----------
    normal_path : str
        正常人数据路径
    stroke_path : str
        中风患者数据路径
    apply_zscore : bool
        是否进行 Z-score 归一化（默认 False）
    val_ratio : float
        验证集占比（按受试者数）
    random_seed : int
        随机种子

    Returns
    -------
    dict with keys:
        train_dataset, val_dataset,
        train_loader, val_loader,
        zscore_mean, zscore_std  (如未使用则为 None)
    """
    normal = _load_pkl(normal_path)
    stroke = _load_pkl(stroke_path)

    all_samples = []   # 每个元素 (F, M, D) = (1001, 28, 3)
    all_labels = []    # 0 = normal, 1 = stroke
    all_groups = []    # 受试者唯一 ID（用于 group split）

    filtered_raw_invalid = 0
    filtered_processed_invalid = 0

    group_id = 0

    # ---- 处理正常人 ----
    for person_data in normal["data"]:
        # person_data: (28, n_trials, 1001, 3)  →  需要转为 (n_trials, 1001, 28, 3)
        trials = person_data.transpose(1, 2, 0, 3)   # (n_trials, 1001, 28, 3)
        n_trials = trials.shape[0]
        for t in range(n_trials):
            raw_sample = trials[t]
            if not is_valid_raw_sample(raw_sample):
                filtered_raw_invalid += 1
                continue

            sample = preprocess_sample(raw_sample)     # (1001, 28, 3)
            if not is_valid_processed_sample(sample):
                filtered_processed_invalid += 1
                continue

            all_samples.append(sample)
            all_labels.append(0)
            all_groups.append(group_id)
        group_id += 1

    # ---- 处理中风患者 ----
    for person_data in stroke["data"]:
        trials = person_data.transpose(1, 2, 0, 3)
        n_trials = trials.shape[0]
        for t in range(n_trials):
            raw_sample = trials[t]
            if not is_valid_raw_sample(raw_sample):
                filtered_raw_invalid += 1
                continue

            sample = preprocess_sample(raw_sample)
            if not is_valid_processed_sample(sample):
                filtered_processed_invalid += 1
                continue

            all_samples.append(sample)
            all_labels.append(1)
            all_groups.append(group_id)
        group_id += 1

    if len(all_samples) == 0:
        raise ValueError("过滤后没有可用样本，请检查原始数据质量。")

    X = np.stack(all_samples, axis=0).astype(np.float32)  # (N, 1001, 28, 3)
    y = np.array(all_labels, dtype=np.int64)               # (N,)
    groups = np.array(all_groups, dtype=np.int64)           # (N,)

    print(f"数据总量: {X.shape[0]} 样本 "
          f"(正常={np.sum(y == 0)}, 中风={np.sum(y == 1)})")
    print(
        "过滤样本: "
        f"原始无效={filtered_raw_invalid}, 预处理后无效={filtered_processed_invalid}"
    )
    print(f"受试者总数: {group_id} (正常={len(normal['data'])}, 中风={len(stroke['data'])})")

    # ---- 受试者级别划分训练/验证集 ----
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_seed)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"训练集: {len(X_train)} 样本 "
          f"(正常={np.sum(y_train == 0)}, 中风={np.sum(y_train == 1)}) "
          f"| 受试者: {len(np.unique(groups[train_idx]))}")
    print(f"验证集: {len(X_val)} 样本 "
          f"(正常={np.sum(y_val == 0)}, 中风={np.sum(y_val == 1)}) "
          f"| 受试者: {len(np.unique(groups[val_idx]))}")

    # 验证无数据泄露
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    assert train_groups.isdisjoint(val_groups), "数据泄露！训练集和验证集存在相同受试者"

    # ---- Z-score 归一化 (可选) ----
    zscore_mean, zscore_std = None, None
    if apply_zscore:
        X_train, zscore_mean, zscore_std = zscore_normalize(X_train)
        X_val, _, _ = zscore_normalize(X_val, mean=zscore_mean, std=zscore_std)
        print(f"Z-score 归一化已应用 (mean={zscore_mean}, std={zscore_std})")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "train_groups": groups[train_idx],
        "val_groups": groups[val_idx],
        "zscore_mean": zscore_mean,
        "zscore_std": zscore_std,
    }


# ===================================================================
#  PyTorch Dataset & DataLoader
# ===================================================================

class GaitDataset(Dataset):
    """步态骨骼数据集"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, 1001, 28, 3) float32
        y: (N,) int64
        """
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(
    normal_path: str = "data/normal_LsideSegm_28markers.pkl",
    stroke_path: str = "data/stroke_NsideSegm_28markers.pkl",
    apply_zscore: bool = False,
    val_ratio: float = 0.2,
    batch_size: int = 8,
    num_workers: int = 2,
    random_seed: int = 42,
) -> dict:
    """
    一键获取训练/验证 DataLoader。

    Returns
    -------
    dict with keys:
        train_loader, val_loader,
        train_dataset, val_dataset,
        zscore_mean, zscore_std
    """
    prepared = load_and_prepare(
        normal_path=normal_path,
        stroke_path=stroke_path,
        apply_zscore=apply_zscore,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )

    train_dataset = GaitDataset(prepared["X_train"], prepared["y_train"])
    val_dataset = GaitDataset(prepared["X_val"], prepared["y_val"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "zscore_mean": prepared["zscore_mean"],
        "zscore_std": prepared["zscore_std"],
    }


# ===================================================================
#  快速测试
# ===================================================================
if __name__ == "__main__":
    result = get_dataloaders(batch_size=4, apply_zscore=False)

    train_loader = result["train_loader"]
    val_loader = result["val_loader"]

    for X_batch, y_batch in train_loader:
        print(f"\n训练批次 — X: {X_batch.shape}, y: {y_batch.shape}")
        print(f"  标签分布: {y_batch.tolist()}")
        print(f"  X 值范围: [{X_batch.min():.3f}, {X_batch.max():.3f}]")
        break

    for X_batch, y_batch in val_loader:
        print(f"验证批次 — X: {X_batch.shape}, y: {y_batch.shape}")
        break
