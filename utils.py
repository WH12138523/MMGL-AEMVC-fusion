import os
import random
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    # FUSION MODIFICATION: ensure deterministic behavior for all experiment modes.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float((y_true == y_pred).mean())


def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # FUSION MODIFICATION: lightweight AUC implementation without external dependency.
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = y_prob.argsort().argsort() + 1
    sum_pos_ranks = ranks[pos].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def sen(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    return tp / (tp + fn + 1e-12)


def spe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    return tn / (tn + fp + 1e-12)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_results_csv(rows: List[Dict], output_csv: str) -> None:
    import csv

    ensure_dir(os.path.dirname(output_csv) or ".")
    if not rows:
        return
    fieldnames = sorted(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy_curve(results_rows: List[Dict], output_png: str) -> None:
    # FUSION MODIFICATION: optional plotting utility for method-vs-missing-rate benchmark.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not results_rows:
        return
    ensure_dir(os.path.dirname(output_png) or ".")
    methods = sorted(set(r["method"] for r in results_rows))
    rates = sorted(set(float(r["missing_rate"]) for r in results_rows))
    plt.figure(figsize=(8, 5))
    for m in methods:
        xs, ys = [], []
        for r in rates:
            vals = [float(x["acc"]) for x in results_rows if x["method"] == m and float(x["missing_rate"]) == r]
            if vals:
                xs.append(r)
                ys.append(float(np.mean(vals)))
        if xs:
            plt.plot(xs, ys, marker="o", label=m)
    plt.xlabel("Missing rate")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def plot_ablation_bars(rows: List[Dict], output_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    ensure_dir(os.path.dirname(output_png) or ".")
    labels = [r["ablation"] for r in rows]
    acc = [float(r["acc"]) for r in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, acc)
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def run_tsne_plot(features: np.ndarray, labels: np.ndarray, output_png: str) -> None:
    # FUSION MODIFICATION: t-SNE output artifact; silently skipped if sklearn unavailable.
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except Exception:
        return
    ensure_dir(os.path.dirname(output_png) or ".")
    emb = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(features)
    plt.figure(figsize=(6, 5))
    labels = np.asarray(labels).reshape(-1)
    for cls in sorted(np.unique(labels)):
        idx = labels == cls
        plt.scatter(emb[idx, 0], emb[idx, 1], s=12, label=f"class-{cls}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
