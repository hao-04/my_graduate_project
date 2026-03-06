"""
GCN-BERT 骨骼步态分类 — Optuna 超参数搜索脚本
==================================================
搜索目标:
    最小化验证集损失 val_loss

记录内容:
    - 每个 trial 的最佳 val_acc
    - 每个 epoch / trial 的训练耗时
    - GPU 显存占用（current / peak allocated / peak reserved）

用法示例:
    python train_optuna.py
    python train_optuna.py --n_trials 30 --epochs 60 --apply_zscore
    python train_optuna.py --study_name gait_optuna_v1 --timeout 7200
"""

import argparse
import csv
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import get_dataloaders
from model import GCN_BERT


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("train_optuna")
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


def _gpu_mem_stats_mb(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {
            "gpu_mem_current_alloc_mb": 0.0,
            "gpu_mem_peak_alloc_mb": 0.0,
            "gpu_mem_peak_reserved_mb": 0.0,
        }

    current_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    return {
        "gpu_mem_current_alloc_mb": float(current_alloc),
        "gpu_mem_peak_alloc_mb": float(peak_alloc),
        "gpu_mem_peak_reserved_mb": float(peak_reserved),
    }


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
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
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)
    return loss_meter.avg, float(acc)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        loss_meter.update(loss.item(), X.size(0))
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    return {
        "loss": float(loss_meter.avg),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def suggest_hparams(trial: optuna.trial.Trial) -> Dict[str, float]:
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    params = {
        "gcn_hidden": trial.suggest_categorical("gcn_hidden", [32, 64, 128]),
        "d_model": d_model,
        "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 5),
        "dim_ff": 4 * d_model,
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-5, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
    }
    return params


def create_study(args, save_dir: Path):
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=args.pruner_warmup_epochs,
    )

    storage = f"sqlite:///{save_dir / 'optuna_study.db'}"

    return optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )


def main(args):
    set_seed(args.seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / run_id
    trial_dir = save_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(save_dir / f"optuna_log_{run_id}.txt")
    logger.info("启动 Optuna 搜索, run_id=%s", run_id)

    with open(save_dir / f"optuna_config_{run_id}.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    study = create_study(args, save_dir)

    def objective(trial: optuna.trial.Trial) -> float:
        hp = suggest_hparams(trial)
        trial_index = trial.number
        this_trial_dir = trial_dir / f"trial_{trial_index:04d}"
        this_trial_dir.mkdir(parents=True, exist_ok=True)

        loaders = get_dataloaders(
            normal_path=args.normal_path,
            stroke_path=args.stroke_path,
            apply_zscore=args.apply_zscore,
            val_ratio=args.val_ratio,
            batch_size=hp["batch_size"],
            num_workers=args.num_workers,
            random_seed=args.seed,
        )
        train_loader = loaders["train_loader"]
        val_loader = loaders["val_loader"]

        train_labels = loaders["train_dataset"].y.numpy()
        class_counts = np.bincount(train_labels, minlength=args.num_classes)
        safe_counts = np.where(class_counts == 0, 1, class_counts)
        class_weights = 1.0 / safe_counts.astype(np.float32)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.from_numpy(class_weights).to(device)

        model = GCN_BERT(
            num_classes=args.num_classes,
            gcn_hidden=hp["gcn_hidden"],
            d_model=hp["d_model"],
            nhead=hp["nhead"],
            num_encoder_layers=hp["num_encoder_layers"],
            dim_feedforward=hp["dim_ff"],
            dropout=hp["dropout"],
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=max(hp["lr"] * 0.01, 1e-8))

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        best_val_loss = math.inf
        best_epoch = 0
        best_val_metrics = None
        history = []
        trial_start = time.perf_counter()

        try:
            for epoch in range(1, args.epochs + 1):
                epoch_start = time.perf_counter()

                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_metrics = evaluate(model, val_loader, criterion, device)

                scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
                epoch_time = time.perf_counter() - epoch_start

                mem_stats = _gpu_mem_stats_mb(device)

                epoch_record = {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_metrics["loss"]),
                    "val_acc": float(val_metrics["accuracy"]),
                    "val_precision": float(val_metrics["precision"]),
                    "val_recall": float(val_metrics["recall"]),
                    "val_f1": float(val_metrics["f1"]),
                    "lr": float(lr),
                    "epoch_time_sec": float(epoch_time),
                    **mem_stats,
                }
                history.append(epoch_record)

                logger.info(
                    "Trial %d | Epoch %d/%d | val_loss=%.4f | val_acc=%.4f | t=%.2fs | peak_alloc=%.1fMB",
                    trial_index,
                    epoch,
                    args.epochs,
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    epoch_time,
                    mem_stats["gpu_mem_peak_alloc_mb"],
                )

                trial.report(val_metrics["loss"], step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"pruned at epoch {epoch}")

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = float(val_metrics["loss"])
                    best_epoch = epoch
                    best_val_metrics = val_metrics
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "hparams": hp,
                            "val_metrics": val_metrics,
                        },
                        this_trial_dir / "best_model.pth",
                    )

                if args.patience > 0 and (epoch - best_epoch) >= args.patience:
                    logger.info("Trial %d 触发早停: patience=%d", trial_index, args.patience)
                    break

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned("CUDA out of memory") from exc
            raise

        trial_time_sec = time.perf_counter() - trial_start
        final_mem = _gpu_mem_stats_mb(device)

        trial.set_user_attr("best_epoch", int(best_epoch))
        trial.set_user_attr("best_val_loss", float(best_val_loss))
        trial.set_user_attr("trial_time_sec", float(trial_time_sec))
        trial.set_user_attr("gpu_mem_peak_alloc_mb", float(final_mem["gpu_mem_peak_alloc_mb"]))
        trial.set_user_attr("gpu_mem_peak_reserved_mb", float(final_mem["gpu_mem_peak_reserved_mb"]))

        trial_summary = {
            "trial_number": int(trial_index),
            "hparams": hp,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_metrics": best_val_metrics,
            "trial_time_sec": float(trial_time_sec),
            **final_mem,
        }

        with open(this_trial_dir / "summary.json", "w") as f:
            json.dump(trial_summary, f, indent=2, ensure_ascii=False)

        with open(this_trial_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        return float(best_val_loss)

    start_time = time.perf_counter()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    total_time = time.perf_counter() - start_time

    best_trial = study.best_trial
    logger.info("%s", "=" * 80)
    logger.info("Optuna 搜索完成")
    logger.info("最佳 Trial: %d", best_trial.number)
    logger.info("最佳 val_loss: %.6f", best_trial.value)
    logger.info("最佳超参数: %s", best_trial.params)
    logger.info("总耗时: %.2f 秒", total_time)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    summary = {
        "run_id": run_id,
        "study_name": args.study_name,
        "direction": "minimize",
        "objective": "val_loss",
        "n_trials_requested": int(args.n_trials),
        "n_trials_total": int(len(study.trials)),
        "n_trials_completed": int(len(completed_trials)),
        "n_trials_pruned": int(len(pruned_trials)),
        "total_time_sec": float(total_time),
        "best_trial": {
            "number": int(best_trial.number),
            "value": float(best_trial.value),
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
        },
    }

    with open(save_dir / f"optuna_summary_{run_id}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = save_dir / f"optuna_trials_{run_id}.csv"
    csv_fields = [
        "trial_number",
        "state",
        "value",
        "best_epoch",
        "trial_time_sec",
        "gpu_mem_peak_alloc_mb",
        "gpu_mem_peak_reserved_mb",
        "gcn_hidden",
        "d_model",
        "nhead",
        "num_encoder_layers",
        "dim_ff",
        "dropout",
        "lr",
        "weight_decay",
        "batch_size",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for t in study.trials:
            row = {
                "trial_number": int(t.number),
                "state": str(t.state).split(".")[-1],
                "value": "" if t.value is None else float(t.value),
                "best_epoch": t.user_attrs.get("best_epoch", ""),
                "trial_time_sec": t.user_attrs.get("trial_time_sec", ""),
                "gpu_mem_peak_alloc_mb": t.user_attrs.get("gpu_mem_peak_alloc_mb", ""),
                "gpu_mem_peak_reserved_mb": t.user_attrs.get("gpu_mem_peak_reserved_mb", ""),
                "gcn_hidden": t.params.get("gcn_hidden", ""),
                "d_model": t.params.get("d_model", ""),
                "nhead": t.params.get("nhead", ""),
                "num_encoder_layers": t.params.get("num_encoder_layers", ""),
                "dim_ff": (4 * t.params["d_model"]) if "d_model" in t.params else "",
                "dropout": t.params.get("dropout", ""),
                "lr": t.params.get("lr", ""),
                "weight_decay": t.params.get("weight_decay", ""),
                "batch_size": t.params.get("batch_size", ""),
            }
            writer.writerow(row)

    logger.info("结果目录: %s", save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="GCN-BERT Optuna 超参数搜索")

    parser.add_argument("--normal_path", type=str, default="data/normal_LsideSegm_28markers.pkl")
    parser.add_argument("--stroke_path", type=str, default="data/stroke_NsideSegm_28markers.pkl")
    parser.add_argument("--apply_zscore", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=0, help="秒；0 表示不限时")
    parser.add_argument("--study_name", type=str, default="gcn_bert_optuna")
    parser.add_argument("--pruner_startup_trials", type=int, default=5)
    parser.add_argument("--pruner_warmup_epochs", type=int, default=8)

    parser.add_argument("--save_dir", type=str, default="output/optuna")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.timeout <= 0:
        args.timeout = None

    if args.seed < 0:
        raise ValueError("seed 必须为非负整数")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    main(args)
