#!/usr/bin/env bash
# Usage: tools/run_3d.sh <local_block.zarr> <out_dir>
# Runs the full-3D DINO-guided ink U-Net (256^3 patches) over a local zarr block.
set -euo pipefail
ROOT=/c/Users/mdevi/prize/vesuvius
IN="$1"; OUT="$2"
cd "$ROOT/villa/vesuvius"
TORCH_COMPILE_DISABLE=1 uv run --no-sync python -m vesuvius.models.run.inference \
  --model_path "$ROOT/checkpoints/ink_3d_dino_guided/ckpt_78k_fullsup.pth" \
  --model_type train_py \
  --input_dir "$IN/0" --input_format zarr \
  --output_dir "$OUT" \
  --patch_size 256,256,256 --overlap 0.25 --batch_size 1 \
  --normalization percentile_minmax --save_softmax --disable_tta --verbose "${@:3}"
