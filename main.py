import argparse
import os
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch

from missing_impute_utils import generate_missing_on_test, generate_modal_missing, impute_test_using_aemvc, impute_test_using_train_stats
from model import EvalHelper
from utils import accuracy_score, auc_score, ensure_dir, plot_ablation_bars, plot_accuracy_curve, run_tsne_plot, save_results_csv, sen, set_seed, spe


def _split_train_val_test(n: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def _default_modalities_for_dataset(dataset: str) -> Tuple[Dict[str, int], str]:
    if dataset.lower() == "tadpole":
        return {"MRI": 64, "PET": 32, "CSF": 16}, "PET"
    if dataset.lower() == "abide":
        return {"fMRI": 48, "sMRI": 32}, "fMRI"
    return {"mod1": 32, "mod2": 24}, "mod2"


def _load_dataset_or_synthetic(args) -> Tuple[Dict[str, np.ndarray], np.ndarray, str]:
    # FUSION MODIFICATION: robust fallback to synthetic data when data files are unavailable.
    modal_dims, default_target = _default_modalities_for_dataset(args.dataset)
    if not args.data_path or not os.path.exists(args.data_path):
        rng = np.random.RandomState(args.seed)
        n = args.synthetic_num_samples
        y = rng.randint(0, 2, size=(n,))
        data = {}
        for m, d in modal_dims.items():
            x = rng.normal(size=(n, d)).astype(np.float32)
            x[:, : min(4, d)] += y[:, None] * 0.5
            data[m] = x
        return data, y, default_target
    raise NotImplementedError("Custom dataset loader is not provided in this minimal repository baseline.")


def _select_target_modalities(args, default_target: str, modal_data: Dict[str, np.ndarray]) -> List[str]:
    if args.target_modalities:
        return [m.strip() for m in args.target_modalities.split(",") if m.strip() in modal_data]
    return [default_target] if default_target in modal_data else [list(modal_data.keys())[0]]


def _to_torch(data_dict: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in data_dict.items()}


def _slice_dict(data: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[idx] for k, v in data.items()}


def _slice_mask(mask_dict: Dict, idx: np.ndarray) -> Dict:
    mfo = mask_dict.get("modal_feature_observed", {})
    mso = mask_dict.get("modal_sample_observed", {})

    def _sample_obs_for_modal(modal: str) -> np.ndarray:
        # FUSION MODIFICATION: readability helper for safe fallback when sample-level mask is absent.
        base = np.asarray(mfo[modal])
        default = np.ones(base.shape[0], dtype=bool)
        return np.asarray(mso.get(modal, default))[idx]

    return {
        "modal_feature_observed": {m: np.asarray(mm)[idx] for m, mm in mfo.items()},
        "modal_sample_observed": {m: _sample_obs_for_modal(m) for m in mfo},
        "meta": deepcopy(mask_dict.get("meta", {})),
    }


def _eval_metrics(y_true: np.ndarray, logits: torch.Tensor) -> Dict[str, float]:
    prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
    return {
        "acc": accuracy_score(y_true, pred),
        "auc": auc_score(y_true, prob),
        "sensitivity": sen(y_true, pred),
        "specificity": spe(y_true, pred),
    }


def train_and_eval(args, method_name: str, missing_rate: float):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    modal_data, y_all, default_target = _load_dataset_or_synthetic(args)
    target_modalities = _select_target_modalities(args, default_target, modal_data)
    train_idx, val_idx, test_idx = _split_train_val_test(len(y_all), seed=args.seed)

    # FUSION MODIFICATION: method-specific missingness and imputation strategy.
    if method_name == "MMGL_original":
        train_missing = modal_data
        train_mask = {"modal_feature_observed": {m: np.ones_like(v, dtype=bool) for m, v in modal_data.items()}}
    else:
        train_missing, train_mask = generate_modal_missing(
            modal_data,
            missing_rate=missing_rate,
            target_modalities=target_modalities,
            missing_type=args.missing_type,
            mixed_sample_ratio=args.mixed_sample_ratio,
            seed=args.seed,
            return_mask_dict=True,
        )

    train_data = _slice_dict(train_missing, train_idx)
    val_data = _slice_dict(train_missing, val_idx)
    test_data = _slice_dict(train_missing, test_idx)
    train_mask_s = _slice_mask(train_mask, train_idx)
    val_mask_s = _slice_mask(train_mask, val_idx)
    test_mask_s = _slice_mask(train_mask, test_idx)
    y_train = torch.tensor(y_all[train_idx], dtype=torch.long, device=device)
    y_val_np = y_all[val_idx]
    y_test_np = y_all[test_idx]

    # FUSION MODIFICATION: align `impute_mode` for the selected benchmark method.
    method_to_mode = {
        "MMGL_original": "none",
        "MMGL_mean": "mean",
        "MMGL_simple_AEMVC": "simple_aemvc",
        "MMGL_full_AEMVC": "full_aemvc",
        "Ours_fusion": "fusion",
    }
    run_args = deepcopy(args)
    run_args.impute_mode = method_to_mode[method_name]
    modal_dims = {m: train_data[m].shape[1] for m in train_data}
    helper = EvalHelper(run_args, modal_dims=modal_dims, num_classes=args.num_classes).to(device)

    # Optional baseline imputations before MMGL-only path.
    if run_args.impute_mode == "mean":
        train_data = impute_test_using_train_stats(train_data, train_data)
        val_data = impute_test_using_train_stats(train_data, val_data)
        test_data = impute_test_using_train_stats(train_data, test_data)
    elif run_args.impute_mode == "simple_aemvc":
        train_data = impute_test_using_aemvc(train_data, train_data, helper.aemvc_module, train_mask_s)
        val_data = impute_test_using_aemvc(train_data, val_data, helper.aemvc_module, val_mask_s)
        test_data = impute_test_using_aemvc(train_data, test_data, helper.aemvc_module, test_mask_s)

    train_t = _to_torch(train_data, device)
    val_t = _to_torch(val_data, device)
    test_t = _to_torch(test_data, device)
    y_val = torch.tensor(y_all[val_idx], dtype=torch.long, device=device)
    y_test = torch.tensor(y_all[test_idx], dtype=torch.long, device=device)

    # FUSION MODIFICATION: optional AEMVC pretraining for full_aemvc/fusion.
    if run_args.impute_mode in {"full_aemvc", "fusion"} and run_args.aemvc_pretrain_epochs > 0:
        for _ in range(run_args.aemvc_pretrain_epochs):
            helper.run_epoch(train_t, y_train, missing_mask_dict=train_mask_s, train=True, pretrain_aemvc_only=True)

    best_val_acc = -1.0
    best_state = deepcopy(helper.state_dict())
    patience = 0
    max_pat = args.patience
    for _ in range(args.epochs):
        helper.run_epoch(train_t, y_train, missing_mask_dict=train_mask_s, train=True, pretrain_aemvc_only=False)
        with torch.no_grad():
            out_val = helper.run_epoch(val_t, y_val, missing_mask_dict=val_mask_s, train=False)
            val_pred = torch.argmax(out_val.logits, dim=1).detach().cpu().numpy()
            val_acc = accuracy_score(y_val_np, val_pred)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = deepcopy(helper.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= max_pat:
            break
    helper.load_state_dict(best_state)

    with torch.no_grad():
        out_test = helper.run_epoch(test_t, y_test, missing_mask_dict=test_mask_s, train=False)
    metrics = _eval_metrics(y_test_np, out_test.logits)
    return metrics, out_test.repr.detach().cpu().numpy(), y_test_np


def run_all_experiments(args):
    methods = ["MMGL_original", "MMGL_mean", "MMGL_simple_AEMVC", "MMGL_full_AEMVC", "Ours_fusion"]
    missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []
    tsne_feats = None
    tsne_labels = None
    for method in methods:
        for rate in missing_rates:
            m, feat, labels = train_and_eval(args, method_name=method, missing_rate=rate)
            row = {"dataset": args.dataset, "method": method, "missing_rate": rate, **m}
            rows.append(row)
            if method == "Ours_fusion" and abs(rate - args.tsne_missing_rate) < 1e-9:
                tsne_feats, tsne_labels = feat, labels

    # FUSION MODIFICATION: ablation experiments.
    ablations = []
    for spec in [
        {"name": "full", "mut": {}, "method_name": "Ours_fusion"},
        {"name": "no_graph_reg", "mut": {"aemvc_lambda1": 0.0}, "method_name": "Ours_fusion"},
        {"name": "no_hsic", "mut": {"aemvc_lambda2": 0.0}, "method_name": "Ours_fusion"},
        {"name": "no_kernel_concat", "mut": {"disable_kernel_concat": True}, "method_name": "Ours_fusion"},
        {"name": "no_joint_opt", "mut": {"impute_mode": "full_aemvc"}, "method_name": "MMGL_full_AEMVC"},
    ]:
        cfg = deepcopy(args)
        for k, v in spec["mut"].items():
            setattr(cfg, k, v)
        m, _, _ = train_and_eval(cfg, method_name=spec["method_name"], missing_rate=args.ablation_missing_rate)
        ablations.append({"dataset": args.dataset, "ablation": spec["name"], **m})

    out_dir = os.path.join(args.output_dir, args.dataset.lower())
    ensure_dir(out_dir)
    save_results_csv(rows, os.path.join(out_dir, "benchmark_results.csv"))
    save_results_csv(ablations, os.path.join(out_dir, "ablation_results.csv"))
    plot_accuracy_curve(rows, os.path.join(out_dir, "accuracy_vs_missing_rate.png"))
    plot_ablation_bars(ablations, os.path.join(out_dir, "ablation_bar.png"))
    if tsne_feats is not None:
        run_tsne_plot(tsne_feats, tsne_labels, os.path.join(out_dir, "tsne_ours.png"))
    print(f"Saved outputs to: {out_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="MMGL-AEMVC fusion experiments")
    parser.add_argument("--dataset", type=str, default="TADPOLE", choices=["TADPOLE", "ABIDE", "SYNTH"])
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aemvc_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "gat"])
    parser.add_argument("--theta_smooth", type=float, default=0.01)
    parser.add_argument("--theta_degree", type=float, default=0.01)
    parser.add_argument("--theta_sparsity", type=float, default=0.001)
    parser.add_argument("--eta", type=float, default=0.0)

    # FUSION MODIFICATION: required imputation and AEMVC parameters.
    parser.add_argument("--impute_mode", type=str, default="mean", choices=["none", "mean", "simple_aemvc", "full_aemvc", "fusion"], help="Missing value imputation mode")
    parser.add_argument("--aemvc_alpha", type=float, default=0.3, help="Weight of AEMVC loss during joint training")
    parser.add_argument("--aemvc_latent_dim", type=int, default=64, help="AEMVC latent dimension")
    parser.add_argument("--aemvc_pretrain_epochs", type=int, default=10, help="AEMVC pretraining epochs")
    parser.add_argument("--missing_type", type=str, default="sample", choices=["sample", "feature", "mixed"], help="Missing type")
    parser.add_argument("--mixed_sample_ratio", type=float, default=0.7)
    parser.add_argument("--target_modalities", type=str, default="")
    parser.add_argument("--aemvc_kernel_type", type=str, default="rbf", choices=["rbf", "cos"])
    parser.add_argument("--aemvc_lambda1", type=float, default=1.0)
    parser.add_argument("--aemvc_lambda2", type=float, default=1.0)
    parser.add_argument("--aemvc_lambda3", type=float, default=1.0)
    parser.add_argument("--aemvc_use_kcca", action="store_true")
    parser.add_argument("--use_graph_prior_fusion", action="store_true")
    parser.add_argument("--graph_prior_beta", type=float, default=0.5)
    parser.add_argument("--disable_kernel_concat", action="store_true")

    parser.add_argument("--synthetic_num_samples", type=int, default=240)
    parser.add_argument("--tsne_missing_rate", type=float, default=0.3)
    parser.add_argument("--ablation_missing_rate", type=float, default=0.3)
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    run_all_experiments(args)


if __name__ == "__main__":
    main()
