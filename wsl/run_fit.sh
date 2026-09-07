#!/usr/bin/env bash
# Usage: run_fit.sh <dataset_dir> <out_dir> <z_begin> <z_end> <steps> [sense] [extra_json_overrides]
# Headless spiral fit (WSL). Writes checkpoints/previews under <out_dir>.
set -euo pipefail
DS="$1"; OUT="$2"; ZB="$3"; ZE="$4"; STEPS="$5"; SENSE="${6:-}"; EXTRA="${7:-}"
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/villa/spiral-fitting"
if [ -n "$SENSE" ]; then
  python3 - "$DS" "$SENSE" <<'EOF'
import json, sys
p = sys.argv[1] + '/spiral-scroll.json'; d = json.load(open(p)); d['spiral_outward_sense'] = sys.argv[2]; json.dump(d, open(p, 'w'), indent=1)
EOF
fi
mkdir -p "$OUT"
export FIT_SPIRAL_OUT_DIR="$OUT"
export WANDB_MODE=disabled
export FIT_SPIRAL_CONFIG_OVERRIDES="{\"z_begin\": $ZB, \"z_end\": $ZE, \"optimizer_num_training_steps\": $STEPS, \"input_disable_patches\": true, \"loss_weight_shell_outer\": 0, \"loss_weight_shell_patch_radius\": 0, \"dense_spacing_mode\": \"grad_mag\", \"loss_weight_dense_spacing\": 0 ${EXTRA}}"
echo "overrides: $FIT_SPIRAL_CONFIG_OVERRIDES"
uv run --no-sync python fit_spiral.py --dataset "$DS" --cache "$HOME/spiral_cache" 2>&1 | tee "$OUT/fit.log"
