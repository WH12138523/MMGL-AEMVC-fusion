#!/usr/bin/env bash
set -euo pipefail

# FUSION MODIFICATION: one-click TADPOLE benchmark runner.
python /home/runner/work/MMGL-AEMVC-fusion/MMGL-AEMVC-fusion/main.py \
  --dataset TADPOLE \
  --output_dir /home/runner/work/MMGL-AEMVC-fusion/MMGL-AEMVC-fusion/outputs \
  --missing_type sample \
  --target_modalities PET
