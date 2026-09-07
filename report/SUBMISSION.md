# Progress Prize submission text (draft for the form)

**Title.** vc-windows-tools: single-GPU, Windows-native ink search from published data products

**Open problem addressed.** Data infrastructure and cloud workflows (2026 open
problems): terabyte volumes, tiled inference, reproducibility for contributors
without a Linux cluster. Secondary: makes the official spiral fitter usable
under WSL2 with a documented headless recipe.

**What it does.** From the organizers' public CT volumes and surface
predictions on S3, with no local copy of a volume, it fetches only the chunks a
sheet touches, renders 21- or 62-layer face-on sheet stacks (four renderers:
height fields from surface predictions for 9 um and 2.4 um data, and tifxyz
windings from a fitted spiral), runs the published ink models (flat 9 um,
canonical 2 um, full-3D DINO-guided, DINO ink-likeness), scores the maps, and
sweeps a whole scroll or a whole fitted mesh unattended. One command reproduces
a block-to-ink-map run.

**Validation.** Reproduces the official ink map of Paris 4 segment
20231016151002 with the flat model; the canonical 2 um model through the
height-field renderer fires on 40% of a Paris 4 sheet at a training-config
coordinate.

**Results on eligible scrolls.** Negative but systematic: PHerc1203 (2.4 um)
with three detectors that all fire on Paris 4; PHerc0125 (spiral fit, ~40
windings swept over a 5-8 mm band); PHerc0268 (6 outer-winding blocks). All
scores, previews and logs are in the repository's results folder. Radial sheet
counts give winding-count estimates for the track-ready scrolls, needed to
configure the fitter (Scroll 1's 130 is wrong for PHerc0125, ~60-90).

**Why it helps others.** A contributor with one consumer GPU and Windows can
now test any model on any block of any eligible scroll in minutes, and leave a
sweep running overnight. Every step is a small script with a documented CLI;
formats are the standard OME-Zarr and tifxyz.

**Links.** GitHub: (to be filled after push). License: MIT. Discord handle: (fill).
