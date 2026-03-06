"""
GCN-BERT 骨骼步态分类 — Optuna 超参数搜索脚本
==================================================
搜索目标:
    最小化 5 折交叉验证平均 best val_loss

记录内容:
    - 每个 trial 的每折最佳 val_loss / val_acc
    - 每个 epoch / fold / trial 的训练耗时
    - GPU 显存占用（current / peak allocated / peak reserved）
    - 5 折划分结果（样本索引与受试者ID）

用法示例:
    python train_optuna.py
    python train_optuna.py --n_trials 30 --epochs 60
    python train_optuna.py --study_name gait_optuna_v1 --timeout 7200
"""

import argparse
import csv
import json
import logging
import math
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
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
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
        "apply_zscore": trial.suggest_categorical("apply_zscore", [False, True]),
    }
    return params


def load_all_samples_with_groups(normal_path: str, stroke_path: str):
    with open(normal_path, "rb") as f:
        normal = pickle.load(f)
    with open(stroke_path, "rb") as f:
        stroke = pickle.load(f)

    all_samples, all_labels, all_groups = [], [], []
    filtered_raw_invalid = 0
    filtered_processed_invalid = 0

    group_id = 0
    subject_metadata = []

    for person_idx, person_data in enumerate(normal["data"]):
        subject_metadata.append(
            {
                "subject_id": int(group_id),
                "cohort": "normal",
                "label": 0,
                "person_index_in_cohort": int(person_idx),
            }
        )
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

    for person_idx, person_data in enumerate(stroke["data"]):
        subject_metadata.append(
            {
                "subject_id": int(group_id),
                "cohort": "stroke",
                "label": 1,
                "person_index_in_cohort": int(person_idx),
            }
        )
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
    return X, y, groups, stats, subject_metadata


def build_group_kfold_splits(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int):
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(X, y, groups))


def save_cv_splits(
    splits,
    y: np.ndarray,
    groups: np.ndarray,
    save_path: Path,
    subject_metadata,
):
    subject_metadata_map = {
        int(item["subject_id"]): item for item in subject_metadata
    }

    split_records = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        train_subjects = np.unique(groups[train_idx])
        val_subjects = np.unique(groups[val_idx])
        overlap = np.intersect1d(train_subjects, val_subjects)

        split_records.append(
            {
                "fold": fold_idx,
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "train_subject_ids": train_subjects.tolist(),
                "val_subject_ids": val_subjects.tolist(),
                "train_subjects": [
                    subject_metadata_map[int(sid)] for sid in train_subjects.tolist()
                ],
                "val_subjects": [
                    subject_metadata_map[int(sid)] for sid in val_subjects.tolist()
                ],
                "n_train_samples": int(len(train_idx)),
                "n_val_samples": int(len(val_idx)),
                "n_train_subjects": int(len(train_subjects)),
                "n_val_subjects": int(len(val_subjects)),
                "train_label_counts": {
                    "normal": int(np.sum(y[train_idx] == 0)),
                    "stroke": int(np.sum(y[train_idx] == 1)),
                },
                "val_label_counts": {
                    "normal": int(np.sum(y[val_idx] == 0)),
                    "stroke": int(np.sum(y[val_idx] == 1)),
                },
                "subject_overlap_count": int(len(overlap)),
            }
        )

    with open(save_path, "w") as f:
        json.dump(split_records, f, indent=2, ensure_ascii=False)


def create_optimizer(model: nn.Module, hp: Dict[str, float]):
    optimizer_name = hp["optimizer"]
    if optimizer_name == "Adam":
        return Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    return AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])


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

    X, y, groups, dataset_stats, subject_metadata = load_all_samples_with_groups(
        normal_path=args.normal_path,
        stroke_path=args.stroke_path,
    )
    logger.info(
        "数据总量: %d 样本 (正常=%d, 中风=%d)",
        dataset_stats["n_total_samples"],
        dataset_stats["n_normal_samples"],
        dataset_stats["n_stroke_samples"],
    )
    logger.info(
        "受试者总数: %d (正常=%d, 中风=%d)",
        dataset_stats["n_subjects"],
        dataset_stats["n_normal_subjects"],
        dataset_stats["n_stroke_subjects"],
    )
    logger.info(
        "过滤样本: 原始无效=%d, 预处理后无效=%d",
        dataset_stats["filtered_raw_invalid"],
        dataset_stats["filtered_processed_invalid"],
    )

    if args.k_folds != 5:
        logger.warning("当前设置 k_folds=%d, 按需求建议使用 5 折。", args.k_folds)

    n_subjects = len(np.unique(groups))
    if args.k_folds > n_subjects:
        raise ValueError(f"k_folds={args.k_folds} 大于受试者数 {n_subjects}")

    cv_splits = build_group_kfold_splits(X, y, groups, n_splits=args.k_folds)
    split_save_path = save_dir / f"cv_splits_{run_id}.json"
    save_cv_splits(cv_splits, y, groups, split_save_path, subject_metadata)
    logger.info("已保存 %d 折划分结果: %s", args.k_folds, split_save_path)

    study = create_study(args, save_dir)

    def objective(trial: optuna.trial.Trial) -> float:
        hp = suggest_hparams(trial)
        trial_index = trial.number
        this_trial_dir = trial_dir / f"trial_{trial_index:04d}"
        this_trial_dir.mkdir(parents=True, exist_ok=True)

        fold_results = []
        split_records = []
        trial_start = time.perf_counter()

        try:
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
                train_groups = set(groups[train_idx].tolist())
                val_groups = set(groups[val_idx].tolist())
                overlap = train_groups.intersection(val_groups)
                if overlap:
                    raise ValueError(
                        f"Fold {fold_idx} 存在受试者泄露: overlap={sorted(list(overlap))[:5]}"
                    )

                split_records.append(
                    {
                        "fold": fold_idx,
                        "n_train_samples": int(len(train_idx)),
                        "n_val_samples": int(len(val_idx)),
                        "n_train_subjects": int(len(train_groups)),
                        "n_val_subjects": int(len(val_groups)),
                        "subject_overlap": int(len(overlap)),
                    }
                )

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                zscore_mean, zscore_std = None, None
                if hp["apply_zscore"]:
                    X_train, zscore_mean, zscore_std = zscore_normalize(X_train)
                    X_val, _, _ = zscore_normalize(X_val)

                train_dataset = GaitDataset(X_train, y_train)
                val_dataset = GaitDataset(X_val, y_val)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=hp["batch_size"],
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=hp["batch_size"],
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
                    gcn_hidden=hp["gcn_hidden"],
                    d_model=hp["d_model"],
                    nhead=hp["nhead"],
                    num_encoder_layers=hp["num_encoder_layers"],
                    dim_feedforward=hp["dim_ff"],
                    dropout=hp["dropout"],
                ).to(device)

                criterion = nn.CrossEntropyLoss(weight=class_weights)
                optimizer = create_optimizer(model, hp)
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=args.epochs,
                    eta_min=max(hp["lr"] * 0.01, 1e-8),
                )

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                best_val_loss = math.inf
                best_epoch = 0
                best_val_metrics = None
                history = []
                fold_start = time.perf_counter()

                for epoch in range(1, args.epochs + 1):
                    epoch_start = time.perf_counter()

                    train_loss, train_acc = train_one_epoch(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        device,
                    )
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
                        "Trial %d | Fold %d/%d | Epoch %d/%d | val_loss=%.4f | val_acc=%.4f | t=%.2fs | peak_alloc=%.1fMB",
                        trial_index,
                        fold_idx,
                        args.k_folds,
                        epoch,
                        args.epochs,
                        val_metrics["loss"],
                        val_metrics["accuracy"],
                        epoch_time,
                        mem_stats["gpu_mem_peak_alloc_mb"],
                    )

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
                                "fold": fold_idx,
                                "val_metrics": val_metrics,
                            },
                            this_trial_dir / f"best_model_fold_{fold_idx}.pth",
                        )

                    if args.patience > 0 and (epoch - best_epoch) >= args.patience:
                        logger.info(
                            "Trial %d | Fold %d 触发早停: patience=%d",
                            trial_index,
                            fold_idx,
                            args.patience,
                        )
                        break

                fold_time_sec = time.perf_counter() - fold_start
                fold_mem = _gpu_mem_stats_mb(device)

                fold_summary = {
                    "fold": fold_idx,
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(best_val_loss),
                    "best_val_metrics": best_val_metrics,
                    "fold_time_sec": float(fold_time_sec),
                    "zscore_mean": None if zscore_mean is None else np.asarray(zscore_mean).tolist(),
                    "zscore_std": None if zscore_std is None else np.asarray(zscore_std).tolist(),
                    "gpu_mem_peak_alloc_mb": float(fold_mem["gpu_mem_peak_alloc_mb"]),
                    "gpu_mem_peak_reserved_mb": float(fold_mem["gpu_mem_peak_reserved_mb"]),
                    "history": history,
                }
                fold_results.append(fold_summary)

                with open(this_trial_dir / f"history_fold_{fold_idx}.json", "w") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)

                completed_fold_losses = [fr["best_val_loss"] for fr in fold_results]
                running_mean_loss = float(np.mean(completed_fold_losses))
                trial.report(running_mean_loss, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"pruned at fold {fold_idx}")

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned("CUDA out of memory") from exc
            raise

        if not fold_results:
            raise optuna.TrialPruned("no completed fold")

        fold_best_losses = [fr["best_val_loss"] for fr in fold_results]
        mean_best_val_loss = float(np.mean(fold_best_losses))
        mean_best_val_acc = float(np.mean([fr["best_val_metrics"]["accuracy"] for fr in fold_results]))

        trial_time_sec = time.perf_counter() - trial_start
        final_mem = _gpu_mem_stats_mb(device)

        trial.set_user_attr("cv_folds", int(args.k_folds))
        trial.set_user_attr("mean_best_val_loss", float(mean_best_val_loss))
        trial.set_user_attr("mean_best_val_acc", float(mean_best_val_acc))
        trial.set_user_attr("trial_time_sec", float(trial_time_sec))
        trial.set_user_attr("gpu_mem_peak_alloc_mb", float(final_mem["gpu_mem_peak_alloc_mb"]))
        trial.set_user_attr("gpu_mem_peak_reserved_mb", float(final_mem["gpu_mem_peak_reserved_mb"]))

        trial_summary = {
            "trial_number": int(trial_index),
            "hparams": hp,
            "cv_folds": int(args.k_folds),
            "split_summary": split_records,
            "fold_results": fold_results,
            "mean_best_val_loss": float(mean_best_val_loss),
            "mean_best_val_acc": float(mean_best_val_acc),
            "trial_time_sec": float(trial_time_sec),
            **final_mem,
        }

        with open(this_trial_dir / "summary.json", "w") as f:
            json.dump(trial_summary, f, indent=2, ensure_ascii=False)

        with open(this_trial_dir / "history.json", "w") as f:
            json.dump(
                {
                    "note": "按 fold 保存的历史见 history_fold_*.json",
                    "cv_folds": int(args.k_folds),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return float(mean_best_val_loss)

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
    logger.info("最佳 5 折平均 val_loss: %.6f", best_trial.value)
    logger.info("最佳超参数: %s", best_trial.params)
    logger.info("总耗时: %.2f 秒", total_time)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    summary = {
        "run_id": run_id,
        "study_name": args.study_name,
        "direction": "minimize",
        "objective": "mean_best_val_loss_over_5fold",
        "dataset_stats": dataset_stats,
        "subject_metadata": subject_metadata,
        "k_folds": int(args.k_folds),
        "cv_splits_file": split_save_path.name,
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
        "cv_folds",
        "mean_best_val_loss",
        "mean_best_val_acc",
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
        "optimizer",
        "apply_zscore",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for t in study.trials:
            row = {
                "trial_number": int(t.number),
                "state": str(t.state).split(".")[-1],
                "value": "" if t.value is None else float(t.value),
                "cv_folds": t.user_attrs.get("cv_folds", ""),
                "mean_best_val_loss": t.user_attrs.get("mean_best_val_loss", ""),
                "mean_best_val_acc": t.user_attrs.get("mean_best_val_acc", ""),
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
                "optimizer": t.params.get("optimizer", ""),
                "apply_zscore": t.params.get("apply_zscore", ""),
            }
            writer.writerow(row)

    logger.info("结果目录: %s", save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="GCN-BERT Optuna 超参数搜索")

    parser.add_argument("--normal_path", type=str, default="data/opendata/normal_LsideSegm_28markers.pkl")
    parser.add_argument("--stroke_path", type=str, default="data/opendata/stroke_NsideSegm_28markers.pkl")
    parser.add_argument("--k_folds", type=int, default=5)
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
    parser.add_argument("--pruner_warmup_epochs", type=int, default=2, help="按折数 warmup")

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
