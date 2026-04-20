# MMGL-AEMVC-fusion

This repository now contains an end-to-end **AEMVC + MMGL fusion** pipeline with backward-compatible MMGL behavior.

## Environment

- Python 3.9+
- `numpy`, `torch`
- Optional: `matplotlib`, `scikit-learn` (for plots / t-SNE)

Install:

```bash
pip install numpy torch matplotlib scikit-learn
```

## Main entry

```bash
python main.py --help
```

Key parameters:

- `--impute_mode`: `none | mean | simple_aemvc | full_aemvc | fusion`
- `--aemvc_alpha`: weight for AEMVC loss in joint training
- `--aemvc_latent_dim`: latent dimension
- `--aemvc_pretrain_epochs`: pretrain epochs for AEMVC
- `--missing_type`: `sample | feature | mixed`
- `--mixed_sample_ratio`: sample/feature ratio in mixed mode
- `--aemvc_lambda1/2/3`: graph/HSIC/kernel-completion weights
- `--use_graph_prior_fusion`: enable `A_final = beta * A_mmgl + (1-beta) * A_aemvc`
- `--graph_prior_beta`: initial beta
- `--disable_kernel_concat`: disable raw+kernel concat

## Single run examples

```bash
python main.py --dataset TADPOLE --impute_mode fusion
python main.py --dataset ABIDE --impute_mode mean --missing_type mixed --mixed_sample_ratio 0.7
```

## Full benchmark scripts

```bash
bash run_tadpole_all.sh
bash run_abide_all.sh
```

## Outputs

Outputs are written to:

- `outputs/<dataset>/benchmark_results.csv`
- `outputs/<dataset>/ablation_results.csv`
- `outputs/<dataset>/accuracy_vs_missing_rate.png`
- `outputs/<dataset>/ablation_bar.png`
- `outputs/<dataset>/tsne_ours.png` (when dependencies are available)

Each CSV includes per-run metrics (`acc`, `auc`, `sensitivity`, `specificity`), and plotting utilities aggregate by method and missing rate.
