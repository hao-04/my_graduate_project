"""
GCN-BERT 骨骼步态分类 — 训练脚本
==================================
用法:
    python train.py
    python train.py --epochs 100 --batch_size 16 --lr 1e-4 --apply_zscore
    python train.py --device cuda:1 --gcn_hidden 128 --dim_ff 512
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from data_loader import get_dataloaders
from model import GCN_BERT


# ===================================================================
#  工具函数
# ===================================================================

def set_seed(seed: int):
    """固定随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """追踪均值"""

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


# ===================================================================
#  训练 & 验证
# ===================================================================

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
        # 梯度裁剪，防止 Transformer 梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(loss.item(), X.size(0))
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)
    return loss_meter.avg, acc


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


# ===================================================================
#  主流程
# ===================================================================

def main(args):
    set_seed(args.seed)

    # ---- 输出目录 ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存超参数
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ---- 设备 ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 数据 ----
    loaders = get_dataloaders(
        normal_path=args.normal_path,
        stroke_path=args.stroke_path,
        apply_zscore=args.apply_zscore,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.seed,
    )
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]

    # ---- 类别权重 (处理不平衡) ----
    train_labels = loaders["train_dataset"].y.numpy()
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts.astype(np.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.from_numpy(class_weights).to(device)
    print(f"类别权重: {class_weights.tolist()}")

    # ---- 模型 ----
    model = GCN_BERT(
        num_classes=args.num_classes,
        gcn_hidden=args.gcn_hidden,
        d_model=128,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计={total_params:,}  可训练={trainable_params:,}")

    # ---- 损失 & 优化器 & 调度器 ----
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ---- 训练循环 ----
    best_val_f1 = 0.0
    best_epoch = 0
    history = []

    print(f"\n{'='*70}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} {'Train Acc':>10} | "
          f"{'Val Loss':>10} {'Val Acc':>10} {'Val F1':>10} | {'LR':>10}")
    print(f"{'='*70}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # 记录
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "lr": current_lr,
            "time": elapsed,
        }
        history.append(record)

        # 打印
        print(
            f"{epoch:>6} | "
            f"{train_loss:>10.4f} {train_acc:>10.4f} | "
            f"{val_metrics['loss']:>10.4f} {val_metrics['accuracy']:>10.4f} "
            f"{val_metrics['f1']:>10.4f} | "
            f"{current_lr:>10.6f}  ({elapsed:.1f}s)"
        )

        # 保存最优模型
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": vars(args),
                },
                save_dir / "best_model.pth",
            )
            print(f"       ↑ 新最优! F1={best_val_f1:.4f}")

        # 保存最新 checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                save_dir / "last_checkpoint.pth",
            )

        # Early stopping
        if args.patience > 0 and (epoch - best_epoch) >= args.patience:
            print(f"\n早停! 已有 {args.patience} 个 epoch 未提升。")
            break

    # ---- 训练结束 ----
    print(f"\n{'='*70}")
    print(f"训练完成! 最优 epoch={best_epoch}, Val F1={best_val_f1:.4f}")

    # 加载最佳模型做最终评估
    ckpt = torch.load(save_dir / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    final_metrics = evaluate(model, val_loader, criterion, device)

    print(f"\n最终验证集结果:")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1 Score:  {final_metrics['f1']:.4f}")
    print(f"  混淆矩阵:  {final_metrics['confusion_matrix']}")

    # 保存训练历史
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n所有结果已保存到 {save_dir}/")


# ===================================================================
#  命令行参数
# ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GCN-BERT 步态分类训练")

    # 数据
    parser.add_argument("--normal_path", type=str,
                        default="data/normal_LsideSegm_28markers.pkl")
    parser.add_argument("--stroke_path", type=str,
                        default="data/stroke_NsideSegm_28markers.pkl")
    parser.add_argument("--apply_zscore", action="store_true",
                        help="是否使用 Z-score 归一化")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_classes", type=int, default=2)

    # 模型
    parser.add_argument("--gcn_hidden", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--dim_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.30751624869734645)

    # 训练
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00041463667378761307)
    parser.add_argument("--weight_decay", type=float, default=1.2327891605450794e-05)
    parser.add_argument("--patience", type=int, default=20,
                        help="早停耐心值，0 表示不早停")

    # 其他
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="output/train")
    parser.add_argument("--save_every", type=int, default=10,
                        help="每隔多少 epoch 保存一次 checkpoint")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
