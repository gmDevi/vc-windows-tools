#!/usr/bin/env bash
# Usage: tools/run_infer.sh <segment_dir_under_data> [extra args]
# Runs the pretrained ink_9um hybrid model (both layer directions) on a segment's surface volume.
set -euo pipefail
ROOT=/c/Users/mdevi/prize/vesuvius
SEG="$1"; shift || true
ZARR=$(ls -d "$ROOT/data/$SEG"/surface-volumes/*.zarr | head -1)
NAME=$(basename "$SEG" | cut -c1-14)
OUT="$ROOT/preds/${NAME}.tif"
cd "$ROOT/villa/vesuvius"
uv run --no-sync python -m vesuvius.ink_detection.inference.infer \
  "$ZARR" "$ROOT/checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth" "$OUT" \
  --overlap 0.5 --blend-mode hann --batch-size 4 --direction both --no-compile "$@"
ls -la "$ROOT/preds/" | grep "$NAME"
