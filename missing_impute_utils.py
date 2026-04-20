from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def _to_numpy_dict(modal_data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in modal_data_dict.items():
        out[k] = np.asarray(v, dtype=np.float32).copy()
    return out


def _build_full_observed_mask(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {m: np.ones_like(x, dtype=bool) for m, x in data.items()}


def _sample_level_missing(mask: np.ndarray, rate: float, rng: np.random.RandomState) -> np.ndarray:
    n = mask.shape[0]
    n_miss = int(round(n * rate))
    if n_miss <= 0:
        return mask
    miss_idx = rng.choice(n, size=n_miss, replace=False)
    mask[miss_idx, :] = False
    return mask


def _feature_level_missing(mask: np.ndarray, rate: float, rng: np.random.RandomState) -> np.ndarray:
    total = mask.size
    n_miss = int(round(total * rate))
    if n_miss <= 0:
        return mask
    flat_idx = rng.choice(total, size=n_miss, replace=False)
    mask.reshape(-1)[flat_idx] = False
    return mask


def _mixed_missing(mask: np.ndarray, rate: float, sample_ratio: float, rng: np.random.RandomState) -> np.ndarray:
    # FUSION MODIFICATION: mixed missing mode combines sample-level and feature-level missingness.
    # The realized global missing ratio can slightly differ from `rate` due to overlap between two masks;
    # users should inspect returned masks when strict realized-rate control is required.
    sample_ratio = float(np.clip(sample_ratio, 0.0, 1.0))
    sample_rate = rate * sample_ratio
    feature_rate = rate * (1.0 - sample_ratio)
    mask = _sample_level_missing(mask, sample_rate, rng)
    mask = _feature_level_missing(mask, feature_rate, rng)
    return mask


def _apply_mask(data: Dict[str, np.ndarray], observed_mask: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for m, x in data.items():
        x_new = x.copy()
        x_new[~observed_mask[m]] = np.nan
        out[m] = x_new
    return out


def generate_modal_missing(
    modal_data_dict: Dict[str, np.ndarray],
    missing_rate: float = 0.2,
    target_modalities: Optional[Iterable[str]] = None,
    missing_type: str = "sample",
    mixed_sample_ratio: float = 0.7,
    seed: int = 42,
    return_mask_dict: bool = False,
):
    """
    Generate missing data on selected modalities.

    Compatibility:
    - Keeps previous API behavior by returning only missing data when return_mask_dict=False.
    - New behavior (return_mask_dict=True) returns (missing_data, missing_mask_dict).

    missing_mask_dict schema:
      {
        "modal_feature_observed": {modal: bool[n, d]},
        "modal_sample_observed": {modal: bool[n]},
        "meta": {...}
      }
    """
    # FUSION MODIFICATION: unified API supporting multi-modality and mixed missingness.
    data = _to_numpy_dict(modal_data_dict)
    modalities = list(data.keys())
    targets = set(target_modalities) if target_modalities else set(modalities)
    rng = np.random.RandomState(seed)
    observed_mask = _build_full_observed_mask(data)

    for m in modalities:
        if m not in targets:
            continue
        mask = observed_mask[m]
        if missing_type == "sample":
            mask = _sample_level_missing(mask, missing_rate, rng)
        elif missing_type == "feature":
            mask = _feature_level_missing(mask, missing_rate, rng)
        elif missing_type == "mixed":
            mask = _mixed_missing(mask, missing_rate, mixed_sample_ratio, rng)
        else:
            raise ValueError(f"Unknown missing_type={missing_type}")
        observed_mask[m] = mask

    missing_data = _apply_mask(data, observed_mask)
    missing_mask_dict = {
        "modal_feature_observed": observed_mask,
        "modal_sample_observed": {m: observed_mask[m].all(axis=1) for m in modalities},
        "meta": {
            "missing_rate": float(missing_rate),
            "missing_type": missing_type,
            "mixed_sample_ratio": float(mixed_sample_ratio),
            "target_modalities": list(targets),
            "seed": int(seed),
        },
    }
    if return_mask_dict:
        return missing_data, missing_mask_dict
    return missing_data


def generate_missing_on_test(
    test_modal_data_dict: Dict[str, np.ndarray],
    missing_rate: float = 0.2,
    target_modalities: Optional[Iterable[str]] = None,
    missing_type: str = "sample",
    mixed_sample_ratio: float = 0.7,
    seed: int = 42,
    return_mask_dict: bool = False,
):
    # FUSION MODIFICATION: test-set missing generator with same unified API for reproducibility.
    return generate_modal_missing(
        modal_data_dict=test_modal_data_dict,
        missing_rate=missing_rate,
        target_modalities=target_modalities,
        missing_type=missing_type,
        mixed_sample_ratio=mixed_sample_ratio,
        seed=seed,
        return_mask_dict=return_mask_dict,
    )


def impute_test_using_train_stats(
    train_modal_data_dict: Dict[str, np.ndarray],
    test_modal_data_dict: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Baseline mean-stat imputation."""
    out = {}
    for m in test_modal_data_dict:
        train_x = np.asarray(train_modal_data_dict[m], dtype=np.float32)
        test_x = np.asarray(test_modal_data_dict[m], dtype=np.float32).copy()
        col_mean = np.nanmean(train_x, axis=0)
        # FUSION MODIFICATION: fallback for columns entirely NaN in training stats.
        global_mean = np.nanmean(train_x)
        if np.isnan(global_mean):
            global_mean = 0.0
        col_mean = np.where(np.isnan(col_mean), global_mean, col_mean)
        nan_idx = np.isnan(test_x)
        if nan_idx.any():
            test_x[nan_idx] = np.take(col_mean, np.where(nan_idx)[1])
        out[m] = test_x
    return out


def impute_test_using_aemvc(
    train_modal_data_dict: Dict[str, np.ndarray],
    test_modal_data_dict: Dict[str, np.ndarray],
    aemvc_module,
    missing_mask_dict: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    # FUSION MODIFICATION: compatibility helper preserving expected "simple_aemvc" behavior.
    if aemvc_module is None:
        return impute_test_using_train_stats(train_modal_data_dict, test_modal_data_dict)
    import torch

    modal_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in test_modal_data_dict.items()}
    if missing_mask_dict is None:
        observed = {k: ~np.isnan(v) for k, v in test_modal_data_dict.items()}
        missing_mask_dict = {"modal_feature_observed": observed}
    with torch.no_grad():
        imputed, _, _ = aemvc_module(
            modal_tensors,
            missing_mask_dict,
        )
    return {k: v.detach().cpu().numpy() for k, v in imputed.items()}
