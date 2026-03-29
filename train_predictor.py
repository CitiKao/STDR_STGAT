"""
train_predictor.py — STGAT 預測模型訓練腳本

訓練流程：
  1. 生成或載入時空資料
  2. 建構 SpatioTemporalDataset 與 DataLoader
  3. 初始化 STGATPredictor
  4. 多任務損失訓練：L = λ₁·L_demand + λ₂·L_supply + λ₃·L_speed
  5. 記錄各項指標並儲存最佳模型

使用方式（需先準備 data/ 真實路網與 build_speed_features 產物）：
    python train_predictor.py
    python train_predictor.py --epochs 100 --device auto --precision bf16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data_loader import SpatioTemporalDataset, load_nyc_real_graph_features
from stgat_model import STGATPredictor


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 CUDA，但目前環境沒有可用 GPU。")
    return device


def resolve_precision(device: torch.device, precision_arg: str) -> str:
    if device.type != "cuda":
        return "fp32"
    if precision_arg == "auto":
        return "bf16"
    return precision_arg


def resolve_num_workers(requested: int, device: torch.device) -> int:
    if requested >= 0:
        return requested
    if device.type == "cuda":
        return min(8, os.cpu_count() or 1)
    return 0


def configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def load_time_meta_for_training(data_dir: str | Path, num_time_steps: int) -> pd.DataFrame:
    time_meta_path = Path(data_dir) / "time_meta.csv"
    if not time_meta_path.exists():
        raise FileNotFoundError(f"找不到 {time_meta_path}，請先執行 build_speed_features.py")
    time_meta = pd.read_csv(time_meta_path)
    if len(time_meta) < num_time_steps:
        raise ValueError(
            f"time_meta.csv 行數 {len(time_meta)} 少於目前資料時間步 {num_time_steps}"
        )
    time_meta = time_meta.iloc[:num_time_steps].copy()
    time_meta["date"] = pd.to_datetime(time_meta["date"], errors="coerce")
    if time_meta["date"].isna().any():
        raise ValueError("time_meta.csv 的 date 欄位有無法解析的值")
    return time_meta


def assign_calendar_split(time_meta: pd.DataFrame) -> pd.Series:
    day = time_meta["date"].dt.day
    return pd.Series(
        np.where(day <= 20, "train", np.where(day <= 24, "val", "test")),
        index=time_meta.index,
        dtype="object",
    )


def build_monthly_split_indices(
    time_meta: pd.DataFrame,
    hist_len: int,
    pred_horizon: int,
) -> dict[str, list[int]]:
    total = len(time_meta) - hist_len - pred_horizon + 1
    split_labels = assign_calendar_split(time_meta)
    splits = {"train": [], "val": [], "test": []}

    for idx in range(max(total, 0)):
        t_start = idx + hist_len
        t_end = t_start + pred_horizon
        target_labels = split_labels.iloc[t_start:t_end].unique()
        if len(target_labels) != 1:
            continue
        splits[str(target_labels[0])].append(idx)

    return splits


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    non_blocking: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    mse: nn.Module,
    lam1: float,
    lam2: float,
    lam3: float,
) -> dict[str, float]:
    losses = {"demand": 0.0, "supply": 0.0, "speed": 0.0, "total": 0.0}
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            node_seq = batch["node_seq"].to(device, non_blocking=non_blocking)
            speed_seq = batch["speed_seq"].to(device, non_blocking=non_blocking)
            d_tgt = batch["demand_target"].to(device, non_blocking=non_blocking)
            c_tgt = batch["supply_target"].to(device, non_blocking=non_blocking)
            v_tgt = batch["speed_target"].to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                d_pred, c_pred, v_pred = model(node_seq, speed_seq)

                loss_d = mse(d_pred, d_tgt)
                loss_c = mse(c_pred, c_tgt)
                loss_v = mse(v_pred, v_tgt)
                loss = lam1 * loss_d + lam2 * loss_c + lam3 * loss_v

            losses["demand"] += loss_d.item()
            losses["supply"] += loss_c.item()
            losses["speed"] += loss_v.item()
            losses["total"] += loss.item()
            n_batches += 1

    for key in losses:
        losses[key] /= max(n_batches, 1)
    return losses


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    precision = resolve_precision(device, args.precision)
    amp_enabled = device.type == "cuda" and precision == "bf16"
    amp_dtype = torch.bfloat16 if amp_enabled else None
    num_workers = resolve_num_workers(args.num_workers, device)
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    # ── 資料準備 ──
    print("載入紐約真實路網與 build_speed_features 特徵 ...")
    data = load_nyc_real_graph_features(
        args.data_dir,
        max_time_steps=args.max_time_steps,
        edge_length_source=args.edge_length_source,
        add_time_features=not args.disable_time_features,
    )

    adj = data["adj"]
    edge_index = data["edge_index"]
    edge_lengths = data["edge_lengths"]
    node_feat = data["node_features"]
    edge_speeds = data["edge_speeds"]
    time_feature_names = data.get("time_feature_names", [])

    N = adj.shape[0]
    nE = edge_index.shape[0]
    t_steps = int(node_feat.shape[0])
    node_feat_dim = int(node_feat.shape[-1])
    print(f"  節點={N}, 邊={nE}, 時間步={t_steps}, node_feat_dim={node_feat_dim}")
    if time_feature_names:
        print(f"  時間特徵: {', '.join(time_feature_names)}")

    # Dataset
    full_ds = SpatioTemporalDataset(
        node_feat, edge_speeds,
        hist_len=args.hist_len,
        pred_horizon=args.pred_horizon,
    )
    time_meta = load_time_meta_for_training(args.data_dir, t_steps)
    split_indices = build_monthly_split_indices(time_meta, args.hist_len, args.pred_horizon)
    train_ds = Subset(full_ds, split_indices["train"])
    val_ds = Subset(full_ds, split_indices["val"])
    test_ds = Subset(full_ds, split_indices["test"])

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print(
        f"  切分方式: 每月 1-20 train / 21-24 val / 25+ test"
    )
    print(
        f"  訓練集={len(train_ds)}, 驗證集={len(val_ds)}, 測試集={len(test_ds)}"
    )
    print(
        f"  裝置={device} | precision={precision} | "
        f"num_workers={num_workers} | pin_memory={pin_memory}"
    )

    # ── 模型 ──
    model = STGATPredictor(
        num_nodes=N,
        edge_index=torch.from_numpy(edge_index),
        edge_lengths=torch.from_numpy(edge_lengths),
        adj_matrix=torch.from_numpy(adj),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_st_blocks=args.num_st_blocks,
        num_gtcn_layers=args.num_gtcn_layers,
        kernel_size=args.kernel_size,
        pred_horizon=args.pred_horizon,
        node_feat_dim=node_feat_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型參數量: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    mse = nn.MSELoss()

    # ── 訓練 ──
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val_loss = float("inf")

    lam1, lam2, lam3 = args.lambda1, args.lambda2, args.lambda3
    print(f"\n開始訓練 | Epochs={args.epochs} | λ=({lam1},{lam2},{lam3})")
    print("-" * 70)

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        train_losses = {"demand": 0.0, "supply": 0.0, "speed": 0.0, "total": 0.0}
        n_batches = 0

        for batch in train_loader:
            node_seq = batch["node_seq"].to(device, non_blocking=non_blocking)       # (B, N, h, C)
            speed_seq = batch["speed_seq"].to(device, non_blocking=non_blocking)     # (B, |E|, h)
            d_tgt = batch["demand_target"].to(device, non_blocking=non_blocking)
            c_tgt = batch["supply_target"].to(device, non_blocking=non_blocking)
            v_tgt = batch["speed_target"].to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                d_pred, c_pred, v_pred = model(node_seq, speed_seq)

                loss_d = mse(d_pred, d_tgt)
                loss_c = mse(c_pred, c_tgt)
                loss_v = mse(v_pred, v_tgt)
                loss = lam1 * loss_d + lam2 * loss_c + lam3 * loss_v

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses["demand"] += loss_d.item()
            train_losses["supply"] += loss_c.item()
            train_losses["speed"] += loss_v.item()
            train_losses["total"] += loss.item()
            n_batches += 1

        for k in train_losses:
            train_losses[k] /= max(n_batches, 1)

        # ── val ──
        model.eval()
        val_losses = evaluate_loader(
            model,
            val_loader,
            device=device,
            non_blocking=non_blocking,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            mse=mse,
            lam1=lam1,
            lam2=lam2,
            lam3=lam3,
        )

        scheduler.step(val_losses["total"])

        # ── 記錄 ──
        elapsed = time.time() - t0
        record = {
            "epoch": epoch,
            "train_total": round(train_losses["total"], 5),
            "train_demand": round(train_losses["demand"], 5),
            "train_supply": round(train_losses["supply"], 5),
            "train_speed": round(train_losses["speed"], 5),
            "val_total": round(val_losses["total"], 5),
            "val_demand": round(val_losses["demand"], 5),
            "val_supply": round(val_losses["supply"], 5),
            "val_speed": round(val_losses["speed"], 5),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed": round(elapsed, 1),
        }
        history.append(record)

        if epoch % args.log_interval == 0 or epoch == 1:
            print(
                f"[Ep {epoch:>4d}]  "
                f"Train={train_losses['total']:.4f} "
                f"(D={train_losses['demand']:.3f} C={train_losses['supply']:.3f} V={train_losses['speed']:.3f})  "
                f"Val={val_losses['total']:.4f}  "
                f"({elapsed:.0f}s)"
            )

        # ── 儲存最佳 ──
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(model.state_dict(), log_dir / "stgat_best.pt")

    # ── 結束 ──
    torch.save(model.state_dict(), log_dir / "stgat_final.pt")
    print(f"\n訓練完成 | Best Val Loss = {best_val_loss:.5f}")
    print(f"模型已儲存至 {log_dir / 'stgat_best.pt'}")

    best_state = torch.load(log_dir / "stgat_best.pt", map_location=device)
    model.load_state_dict(best_state)
    model.eval()
    test_losses = evaluate_loader(
        model,
        test_loader,
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        mse=mse,
        lam1=lam1,
        lam2=lam2,
        lam3=lam3,
    )
    print(
        "最終 Test="
        f"{test_losses['total']:.4f} "
        f"(D={test_losses['demand']:.3f} C={test_losses['supply']:.3f} V={test_losses['speed']:.3f})"
    )

    # 儲存訓練資料元資訊（供 pipeline 使用）
    meta = {
        "num_nodes": N,
        "num_edges": nE,
        "edge_index": edge_index.tolist(),
        "edge_lengths": edge_lengths.tolist(),
        "adj": adj.tolist(),
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_st_blocks": args.num_st_blocks,
        "num_gtcn_layers": args.num_gtcn_layers,
        "kernel_size": args.kernel_size,
        "pred_horizon": args.pred_horizon,
        "hist_len": args.hist_len,
        "node_feat_dim": node_feat_dim,
        "use_time_features": bool(time_feature_names),
        "time_feature_names": time_feature_names,
        "split_strategy": "per_month_day_1_20_train_21_24_val_25_plus_test",
        "split_counts": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "data_source": "nyc_real",
        "data_dir": args.data_dir,
    }
    with open(log_dir / "stgat_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(log_dir / "predictor_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"訓練日誌已儲存至 {log_dir / 'predictor_log.json'}")

    with open(log_dir / "predictor_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_losses, f, ensure_ascii=False, indent=2)
    print(f"測試指標已儲存至 {log_dir / 'predictor_test_metrics.json'}")


# ── CLI ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train STGAT Predictor")

    # 資料
    p.add_argument("--data-dir", type=str, default="data", help="真實資料目錄（adjacency、時序特徵）")
    p.add_argument(
        "--max-time-steps",
        type=int,
        default=0,
        help="截斷時間步（0=全部；除錯或減輕顯存時可設小於完整 T）",
    )
    p.add_argument(
        "--edge-length-source",
        type=str,
        default="osrm",
        choices=["osrm", "centroid"],
        help="邊長來源：osrm 優先讀 edge_lengths_osrm.npy，否則 centroid",
    )
    p.add_argument(
        "--disable-time-features",
        action="store_true",
        help="停用 month / weekday / slot 時間特徵，回到舊版 demand+supply 節點輸入",
    )
    p.add_argument("--hist-len", type=int, default=12, help="歷史窗口 h")
    p.add_argument("--pred-horizon", type=int, default=3, help="預測步數 p")

    # 模型
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-st-blocks", type=int, default=2)
    p.add_argument("--num-gtcn-layers", type=int, default=2)
    p.add_argument("--kernel-size", type=int, default=3)

    # 訓練
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda1", type=float, default=0.4, help="需求損失權重")
    p.add_argument("--lambda2", type=float, default=0.3, help="空車損失權重")
    p.add_argument("--lambda3", type=float, default=0.3, help="速度損失權重")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp32"],
        help="CUDA 上預設 auto->bf16；CPU 會自動退回 fp32",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers；-1 代表自動（CUDA 預設最多 8，CPU 預設 0）",
    )
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--log-interval", type=int, default=5)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
