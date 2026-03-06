"""
GCN-BERT 骨骼步态分类 — GroupKFold 交叉验证训练脚本
==================================================
用法:
    python train_kfold.py
    python train_kfold.py --k_folds 5 --epochs 80 --batch_size 16 --apply_zscore
"""

import argparse
import logging
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data_loader.gait_dataset import (
    GaitDataset,
    is_valid_processed_sample,
    is_valid_raw_sample,
    preprocess_sample,
    zscore_normalize,
)
from model import GCN_BERT


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("train_kfold")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(loss.item(), X.size(0))
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary", zero_division=0)
    recall = recall_score(labels, preds, average="binary", zero_division=0)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    return {
        "loss": loss_meter.avg,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        loss_meter.update(loss.item(), X.size(0))
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    metrics = {
        "loss": loss_meter.avg,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    return metrics


def load_all_samples_with_groups(normal_path: str, stroke_path: str):
    with open(normal_path, "rb") as f:
        normal = pickle.load(f)
    with open(stroke_path, "rb") as f:
        stroke = pickle.load(f)

    all_samples, all_labels, all_groups = [], [], []
    filtered_raw_invalid = 0
    filtered_processed_invalid = 0

    group_id = 0

    for person_data in normal["data"]:
        trials = person_data.transpose(1, 2, 0, 3)
        for trial in trials:
            if not is_valid_raw_sample(trial):
                filtered_raw_invalid += 1
                continue
            sample = preprocess_sample(trial)
            if not is_valid_processed_sample(sample):
                filtered_processed_invalid += 1
                continue

            all_samples.append(sample)
            all_labels.append(0)
            all_groups.append(group_id)
        group_id += 1

    for person_data in stroke["data"]:
        trials = person_data.transpose(1, 2, 0, 3)
        for trial in trials:
            if not is_valid_raw_sample(trial):
                filtered_raw_invalid += 1
                continue
            sample = preprocess_sample(trial)
            if not is_valid_processed_sample(sample):
                filtered_processed_invalid += 1
                continue

            all_samples.append(sample)
            all_labels.append(1)
            all_groups.append(group_id)
        group_id += 1

    if len(all_samples) == 0:
        raise ValueError("过滤后没有可用样本，请检查原始数据质量。")

    X = np.stack(all_samples, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)
    groups = np.array(all_groups, dtype=np.int64)

    stats = {
        "n_total_samples": int(X.shape[0]),
        "n_normal_samples": int(np.sum(y == 0)),
        "n_stroke_samples": int(np.sum(y == 1)),
        "n_subjects": int(group_id),
        "n_normal_subjects": int(len(normal["data"])),
        "n_stroke_subjects": int(len(stroke["data"])),
        "filtered_raw_invalid": int(filtered_raw_invalid),
        "filtered_processed_invalid": int(filtered_processed_invalid),
    }
    return X, y, groups, stats


def _find_latest_optuna_run(optuna_root: Path) -> Path:
    if not optuna_root.exists():
        raise FileNotFoundError(f"Optuna 目录不存在: {optuna_root}")

    run_dirs = [p for p in optuna_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"在 {optuna_root} 下未找到 Optuna 运行目录")

    # 运行目录命名为 YYYYMMDD_HHMMSS，按字典序可得到最新目录
    run_dirs.sort(key=lambda p: p.name)
    return run_dirs[-1]


def _load_optuna_summary(optuna_run_dir: Path) -> dict:
    summary_files = sorted(optuna_run_dir.glob("optuna_summary_*.json"))
    if not summary_files:
        raise FileNotFoundError(f"未找到 optuna_summary_*.json: {optuna_run_dir}")
    with open(summary_files[-1], "r") as f:
        return json.load(f)


def _load_cv_splits(optuna_run_dir: Path, summary: dict):
    cv_splits_file = summary.get("cv_splits_file")
    if cv_splits_file:
        cv_splits_path = optuna_run_dir / cv_splits_file
    else:
        split_files = sorted(optuna_run_dir.glob("cv_splits_*.json"))
        if not split_files:
            raise FileNotFoundError(f"未找到 cv_splits_*.json: {optuna_run_dir}")
        cv_splits_path = split_files[-1]

    with open(cv_splits_path, "r") as f:
        raw_splits = json.load(f)

    splits = []
    for rec in raw_splits:
        train_idx = np.asarray(rec["train_indices"], dtype=np.int64)
        val_idx = np.asarray(rec["val_indices"], dtype=np.int64)
        splits.append((train_idx, val_idx))
    return splits, cv_splits_path


def _apply_optuna_best_params(args, best_params: dict):
    args.gcn_hidden = int(best_params["gcn_hidden"])
    args.d_model = int(best_params["d_model"])
    args.nhead = int(best_params["nhead"])
    args.num_encoder_layers = int(best_params["num_encoder_layers"])
    args.dim_ff = 4 * args.d_model
    args.dropout = float(best_params["dropout"])
    args.lr = float(best_params["lr"])
    args.weight_decay = float(best_params["weight_decay"])
    args.batch_size = int(best_params["batch_size"])
    args.optimizer = str(best_params.get("optimizer", "AdamW"))
    args.apply_zscore = bool(best_params.get("apply_zscore", False))


def _create_optimizer(model: nn.Module, args):
    if args.optimizer == "Adam":
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def run_single_fold(fold_idx, train_idx, val_idx, X, y, groups, args, device, save_dir: Path):
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    overlap = train_groups.intersection(val_groups)
    assert len(overlap) == 0, f"Fold {fold_idx}: 检测到同一受试者跨训练/验证集污染"

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    zscore_mean, zscore_std = None, None
    if args.apply_zscore:
        X_train, zscore_mean, zscore_std = zscore_normalize(X_train)
        X_val, _, _ = zscore_normalize(X_val)

    train_dataset = GaitDataset(X_train, y_train)
    val_dataset = GaitDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    class_counts = np.bincount(y_train, minlength=args.num_classes)
    safe_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / safe_counts.astype(np.float32)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.from_numpy(class_weights).to(device)

    model = GCN_BERT(
        num_classes=args.num_classes,
        gcn_hidden=args.gcn_hidden,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = _create_optimizer(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_f1 = -1.0
    best_epoch = 0
    history = []

    fold_save_dir = save_dir / f"fold_{fold_idx}"
    fold_save_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_kfold")
    logger.info("%s", "=" * 80)
    logger.info("Fold %d/%d", fold_idx, args.k_folds)
    logger.info(
        "样本信息 | train=%d val=%d | 受试者 train=%d val=%d | overlap=%d",
        len(train_idx),
        len(val_idx),
        len(train_groups),
        len(val_groups),
        len(overlap),
    )
    logger.info(
        "类别统计(train) | normal=%d stroke=%d",
        int(np.sum(y_train == 0)),
        int(np.sum(y_train == 1)),
    )

    logger.info(
        "%6s | %10s %10s %10s %10s %10s | %10s %10s %10s %10s %10s | %10s",
        "Epoch",
        "TrLoss",
        "TrAcc",
        "TrPrec",
        "TrRec",
        "TrF1",
        "VaLoss",
        "VaAcc",
        "VaPrec",
        "VaRec",
        "VaF1",
        "LR",
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "lr": current_lr,
            "time": elapsed,
        }
        history.append(record)

        logger.info(
            "%6d | %10.4f %10.4f %10.4f %10.4f %10.4f | %10.4f %10.4f %10.4f %10.4f %10.4f | %10.6f (%.1fs)",
            epoch,
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["precision"],
            train_metrics["recall"],
            train_metrics["f1"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
            current_lr,
            elapsed,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                fold_save_dir / f"best_model_{args.run_id}.pth",
            )
            logger.info("Fold %d 新最优: epoch=%d, val_f1=%.4f", fold_idx, epoch, best_val_f1)

        if args.patience > 0 and (epoch - best_epoch) >= args.patience:
            logger.info("Fold %d 触发早停: patience=%d", fold_idx, args.patience)
            break

    ckpt = torch.load(
        fold_save_dir / f"best_model_{args.run_id}.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    final_metrics = evaluate(model, val_loader, criterion, device)

    with open(fold_save_dir / f"history_{args.run_id}.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    plot_training_curves(
        history=history,
        save_path=fold_save_dir / f"metrics_curve_{args.run_id}.png",
        fold_idx=fold_idx,
    )

    fold_result = {
        "fold": fold_idx,
        "n_train_samples": int(len(train_idx)),
        "n_val_samples": int(len(val_idx)),
        "n_train_subjects": int(len(train_groups)),
        "n_val_subjects": int(len(val_groups)),
        "subject_overlap": int(len(overlap)),
        "best_epoch": int(best_epoch),
        "metrics": final_metrics,
    }

    if args.apply_zscore:
        fold_result["zscore_mean"] = zscore_mean.tolist()
        fold_result["zscore_std"] = zscore_std.tolist()

    return fold_result


def plot_training_curves(history, save_path: Path, fold_idx: int):
    epochs = [h["epoch"] for h in history]

    metric_pairs = [
        ("loss", "train_loss", "val_loss"),
        ("accuracy", "train_acc", "val_acc"),
        ("precision", "train_precision", "val_precision"),
        ("recall", "train_recall", "val_recall"),
        ("f1", "train_f1", "val_f1"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (title, train_key, val_key) in enumerate(metric_pairs):
        ax = axes[i]
        ax.plot(epochs, [h[train_key] for h in history], label="Train", linewidth=2)
        ax.plot(epochs, [h[val_key] for h in history], label="Val/Test", linewidth=2)
        ax.set_title(f"Fold {fold_idx} - {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="GCN-BERT GroupKFold 交叉验证训练")

    parser.add_argument("--normal_path", type=str, default="data/opendata/normal_LsideSegm_28markers.pkl")
    parser.add_argument("--stroke_path", type=str, default="data/opendata/stroke_NsideSegm_28markers.pkl")
    parser.add_argument("--apply_zscore", action="store_true", help="是否使用 Z-score 归一化（若加载 Optuna 最优参数会被覆盖）")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--gcn_hidden", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "AdamW"])
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="output/kfold")
    parser.add_argument("--optuna_root", type=str, default="output/optuna")
    parser.add_argument("--optuna_run_dir", type=str, default="", help="指定 Optuna 运行目录；为空时自动取最新")
    parser.add_argument("--no_use_optuna_best", action="store_true", help="不加载 Optuna 最佳参数与划分")

    return parser.parse_args()


def main(args):
    set_seed(args.seed)

    args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = Path(args.save_dir) / args.run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(save_dir / f"train_log_{args.run_id}.txt")
    logger.info("启动 GroupKFold 训练, run_id=%s", args.run_id)

    with open(save_dir / f"config_{args.run_id}.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    optuna_summary = None
    cv_splits_from_optuna = None
    cv_splits_path = None
    selected_optuna_run = None

    if not args.no_use_optuna_best:
        if args.optuna_run_dir.strip():
            selected_optuna_run = Path(args.optuna_run_dir)
        else:
            selected_optuna_run = _find_latest_optuna_run(Path(args.optuna_root))

        optuna_summary = _load_optuna_summary(selected_optuna_run)
        best_trial = optuna_summary.get("best_trial", {})
        best_params = best_trial.get("params", {})
        if not best_params:
            raise ValueError(f"Optuna summary 中缺少 best_trial.params: {selected_optuna_run}")

        _apply_optuna_best_params(args, best_params)
        cv_splits_from_optuna, cv_splits_path = _load_cv_splits(selected_optuna_run, optuna_summary)
        args.k_folds = len(cv_splits_from_optuna)

        logger.info("使用 Optuna 运行目录: %s", selected_optuna_run)
        logger.info("使用 Optuna 划分文件: %s", cv_splits_path)
        logger.info("使用 Optuna 最佳参数: %s", best_params)

    X, y, groups, stats = load_all_samples_with_groups(
        normal_path=args.normal_path,
        stroke_path=args.stroke_path,
    )

    logger.info(
        f"数据总量: {stats['n_total_samples']} 样本 "
        f"(正常={stats['n_normal_samples']}, 中风={stats['n_stroke_samples']})"
    )
    logger.info(
        f"过滤样本: 原始无效={stats['filtered_raw_invalid']}, "
        f"预处理后无效={stats['filtered_processed_invalid']}"
    )
    logger.info(
        f"受试者总数: {stats['n_subjects']} "
        f"(正常={stats['n_normal_subjects']}, 中风={stats['n_stroke_subjects']})"
    )

    n_subjects = len(np.unique(groups))
    if args.k_folds > n_subjects:
        raise ValueError(f"k_folds={args.k_folds} 大于受试者数 {n_subjects}")

    if cv_splits_from_optuna is not None:
        splitter = cv_splits_from_optuna
        logger.info("按 Optuna 保存的索引划分进行训练，共 %d 折", len(splitter))
    else:
        splitter = list(GroupKFold(n_splits=args.k_folds).split(X, y, groups))
        logger.info("按当前 GroupKFold 重新划分进行训练，共 %d 折", len(splitter))

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter, start=1):
        result = run_single_fold(
            fold_idx=fold_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            X=X,
            y=y,
            groups=groups,
            args=args,
            device=device,
            save_dir=save_dir,
        )
        fold_results.append(result)

        m = result["metrics"]
        logger.info(
            f"Fold {fold_idx} 结果 | "
            f"Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, "
            f"Rec={m['recall']:.4f}, F1={m['f1']:.4f}, "
            f"Overlap={result['subject_overlap']}"
        )

    metric_names = ["accuracy", "precision", "recall", "f1", "loss"]
    summary_metrics = {}
    for name in metric_names:
        values = np.array([fr["metrics"][name] for fr in fold_results], dtype=np.float64)
        summary_metrics[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }

    summary = {
        "dataset_stats": stats,
        "k_folds": int(args.k_folds),
        "optuna_run_dir": "" if selected_optuna_run is None else str(selected_optuna_run),
        "optuna_cv_splits": "" if cv_splits_path is None else str(cv_splits_path),
        "optuna_best_params": {} if optuna_summary is None else optuna_summary["best_trial"]["params"],
        "fold_results": fold_results,
        "summary_metrics": summary_metrics,
    }

    with open(save_dir / f"kfold_summary_{args.run_id}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("%s", "=" * 72)
    logger.info("K 折交叉验证完成")
    logger.info("平均指标:")
    logger.info(
        f"  Accuracy:  {summary_metrics['accuracy']['mean']:.4f} ± {summary_metrics['accuracy']['std']:.4f}\n"
        f"  Precision: {summary_metrics['precision']['mean']:.4f} ± {summary_metrics['precision']['std']:.4f}\n"
        f"  Recall:    {summary_metrics['recall']['mean']:.4f} ± {summary_metrics['recall']['std']:.4f}\n"
        f"  F1 Score:  {summary_metrics['f1']['mean']:.4f} ± {summary_metrics['f1']['std']:.4f}"
    )
    logger.info("结果已保存到: %s", save_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
