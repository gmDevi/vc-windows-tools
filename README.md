# vc-windows-tools: single-GPU, Windows-native ink search for the Vesuvius Challenge

A small, dependency-light toolkit that takes the organizers' published data
products (CT volumes and surface predictions on `s3://vesuvius-challenge-open-data`)
to face-on sheet stacks and ink-probability maps on a single consumer GPU under
Windows, without building VC3D or the Linux-only spiral-fitting stack. It also
includes the glue that makes the official spiral fitter usable under WSL2, and
a mesh renderer that turns its `tifxyz` windings into ink maps from streamed S3
chunks with no local copy of the volume.

Problem statement (from the 2026 open problems): terabyte-scale volumes require
specialized handling and not every contribution is reproducible; many
contributors have one GPU and a laptop, not a cluster. This toolkit lets such a
contributor go from "pick a scroll and a location" to "ink map" in minutes,
with every step scriptable and unattended.

## What is here

| Script | Purpose |
|---|---|
| `tools/fetch_block.py`, `tools/fetch_block3.py` | Resumable download of a cube or box of a level-0 CT volume (uncompressed zarr, 128^3 chunks) |
| `tools/fetch_region_zarr5.py` | Box download from a compressed (blosc) zarr by direct chunk HTTP with retries |
| `tools/heightfield_render.py` (+ `2`, `3`, `4`) | Render 21- or 62-layer sheet stacks by growing per-sheet height fields from the organizers' surface predictions; variants for coarse surfaces, big axis-aligned boxes, and tilted sheets without rotating the volume |
| `tools/render_tifxyz.py` | Render a surface volume from a `tifxyz` mesh (e.g. a fitted spiral winding) by upsampling the grid, estimating normals, and sampling S3 chunks on demand with a local cache |
| `tools/infer_canonical.py` | Run `scrollprize/ink_canonical_2um` on a local stack using the villa `optimized_inference` code |
| `tools/infer3d_compact.py` | In-memory sliding-window inference for the full-3D `ink_3d_dino_guided` model (avoids ~30 GB of per-patch logits) |
| `tools/dino_similarity.py` | Dense DINO ink-likeness (cosine to the organizers' reference ink embedding) with the `dinovol` teacher backbone |
| `tools/sweep.py`, `tools/batch_windings.py` | Unattended sweeps: outer-winding blocks of a scroll, or every winding of a fitted mesh, through the flat model with scoring |
| `tools/score_tif.py`, `tif2png.py`, `montage.py`, `layer_stats.py`, `faceon.py`, `spiral_sense.py` | Scoring, browsing, and diagnostics |
| `wsl/run_fit.sh`, `wsl/fetch_lasagna_slab.py`, `wsl/pget2.sh` | Run the official `spiral-fitting` headlessly under WSL2 on a z-band with only the needed slab of lasagna normals |

## Validation

- Flat 9 um model (`scrollprize/ink_9um`) on a crop of the published Paris 4
  segment `20231016151002` reproduces the organizers' ink map (crisp Greek
  letters, forward direction only). See `report/figures/paris4_ctrl_flat.png`
  next to the official `ink_ds8_crop.png`.
- Canonical 2 um model through `heightfield_render2.py` on a raw Paris 4 block
  plus the L2 surface prediction fires on 40% of a sheet at a location recorded
  in the model's own training config (`sheet3_x515_canon.png`).

## Findings on eligible scrolls (September 2026)

PHerc1203 (the only eligible scroll with a 2.4 um scan) was probed with three
detectors that all fire on Paris 4: the flat model on 21 published segments,
the canonical 2 um model on 1 cm^2 of cleanly rendered sheets, and DINO
ink-likeness. All negative. A spiral fit of PHerc0125 (9.36 um, z 9000-10500)
and a sweep of its outer windings through the flat model were also negative.
Full numbers and figures in `report/REPORT.md`.

## Quick start

```bash
# environment (villa/vesuvius), Windows: skip the C++ package, use the cu130 torch build
uv sync --extra models --no-install-package volume-cartographer
uv pip install torch==2.12.0+cu130 torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install albumentations
set TORCH_COMPILE_DISABLE=1

# 1. a 1024^3 block at 9.36 um plus its surface prediction
python tools/fetch_block.py PHerc1203/volumes/<vol>.zarr/ 8704 3072 1152 1024 data/ct.zarr
python tools/fetch_region_zarr5.py https://.../surfaces/<pred>.zarr/0 8704 3072 1152 1024 1024 1024 data/surf
# 2. render sheets and run the flat model
python tools/heightfield_render.py data/ct.zarr data/surf preds/blk 6
python -m vesuvius.ink_detection.inference.infer preds/blk/sheet0_x123.zarr <ink_9um.pth> preds/blk/sheet0.tif --direction both
```

See `tools/README.md` for per-script usage and `report/REPORT.md` for the
full walkthrough, including the WSL2 spiral-fitting recipe.

License: MIT.
