#!/usr/bin/env bash
# Run inference over every downloaded segment that has no prediction yet.
ROOT=/c/Users/mdevi/prize/vesuvius
for seg in "$ROOT"/data/PHerc0800/segments/* "$ROOT"/data/PHerc1447/segments/*; do
  [ -d "$seg/surface-volumes" ] || continue
  rel=${seg#$ROOT/data/}
  name=$(basename "$seg" | cut -c1-14)
  if [ -f "$ROOT/preds/${name}_reverse.tif" ]; then echo "skip $name"; continue; fi
  echo "=== $rel"
  bash "$ROOT/tools/run_infer.sh" "$rel" 2>&1 | grep -E "^(-rw|Traceback|.*Error)" 
done
