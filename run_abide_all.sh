#!/usr/bin/env bash
set -euo pipefail

# FUSION MODIFICATION: one-click ABIDE benchmark runner.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/main.py" \
  --dataset ABIDE \
  --output_dir "${SCRIPT_DIR}/outputs" \
  --missing_type sample \
  --target_modalities fMRI
