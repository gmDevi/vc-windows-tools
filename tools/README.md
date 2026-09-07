# Windows-native ink-search tools for the Vesuvius Challenge

Lightweight, pure-Python tools to go from the organizers' published data products
(CT volumes + surface predictions on `s3://vesuvius-challenge-open-data`) to
face-on sheet stacks and ink-probability maps, without building VC3D or running
the spiral-fitting stack. Developed on Windows 11 with a single RTX 3090.

## Pipeline

1. **Fetch a block** of a CT volume (uncompressed level-0 zarr, 128^3 chunks) with
   `fetch_block.py` (cube) or `fetch_block3.py` (box). Both are resumable.
2. **Fetch the matching surface-prediction box** (compressed blosc zarr) with
   `fetch_region_zarr5.py` (direct chunk HTTP with retries; robust under load).
3. **Render sheet stacks**:
   - `heightfield_render.py`: 9 um blocks + level-0 surface predictions, 21 layers.
     Rotates the block so sheets are near-vertical, region-grows one height
     field per sheet from surface-prediction peaks, samples layers along the
     normal by linear interpolation.
   - `heightfield_render2.py`: same, but surface prediction may be coarser than the
     CT (e.g. L2 at 9.6 um for a 2.4 um block) and layer count is configurable.
   - `heightfield_render3.py`: big axis-aligned boxes; no rotation, axis
     permutation only.
   - `heightfield_render4.py`: big boxes with tilted sheets; rotates only the small
     surface prediction and samples the original CT through inverse-rotated
     coordinates (no big-volume rotation).
4. **Run an ink model**:
   - flat 9 um model (`scrollprize/ink_9um`) via
     `vesuvius.ink_detection.inference.infer` (`run_infer.sh`, `run_all.sh`).
   - canonical 2 um model (`scrollprize/ink_canonical_2um`, ResNet3D-152 + 3D
     decoder) via `infer_canonical.py`, which reuses
     `villa/ink-detection/optimized_inference` on a local numpy stack.
5. **Score and browse**: `score_tif.py`, `tif2png.py`, `montage.py`,
   `layer_stats.py` (model-free surface-vs-interior views).
6. **Sweep** a scroll unattended with `sweep.py`: picks blocks on the outer
   windings at several heights, runs steps 1 to 5, keeps PNG previews and a
   `summary.json` per block, deletes the raw blocks.

Other helpers: `infer3d_compact.py` (in-memory sliding-window inference for the
full-3D `ink_3d_dino_guided` model, avoiding the ~30 GB per-block logits of the
generic runner), `faceon.py` (quick face-on slab views), `spiral_sense.py`
(estimate spiral handedness from a cross-section), `fetch_sv_crop.py` and
`pool_to_21.py` (crop and z-pool a published 2.4 um surface volume into the
21-slice input the flat model expects).

## Validation

Running the flat model on a crop of the published Paris 4 segment
`20231016151002` (level 2 of its 2.4 um surface volume, pooled to 21 slices)
reproduces the organizers' ink map (crisp Greek letters, forward direction only).
The canonical 2 um model, fed through `heightfield_render2.py` on a raw Paris 4
block plus the L2 surface prediction, fires strongly on a sheet at a location
recorded in the model's own training config.

## Environment

`villa/vesuvius` synced with `uv sync --extra models --no-install-package
volume-cartographer`, then `uv pip install torch==2.12.0+cu130 torchvision
--index-url https://download.pytorch.org/whl/cu130` and `uv pip install
albumentations`. Set `TORCH_COMPILE_DISABLE=1` on Windows (no Triton).
