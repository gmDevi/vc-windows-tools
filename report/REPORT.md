# Single-GPU, Windows-native ink search on the 2025-2026 scans

Gennaro Marco Devincenzis, September 2026. Hardware: one RTX 3090, 64 GB RAM, Windows 11
with WSL2. Everything below ran on that machine within one day, mostly
unattended.

## 1. Problem

The 2026 open problems call out data infrastructure (terabyte volumes, tiled
inference, coordinate metadata) and reproducibility as blockers, and the
First Letters prize targets 13 scrolls scanned in 2025-2026 in which no ink has
been found. Most of the official tooling (VC3D, spiral fitting, Lasagna)
assumes Linux, a C++ toolchain, and many terabytes of local storage. A
contributor with one GPU and a Windows laptop cannot easily ask the simplest
question: "does model X see ink on sheet Y of scroll Z?"

## 2. What was built

A thin toolkit that answers that question from the published data products
alone, with no local copy of a volume.

1. **Fetchers.** Resumable box downloads of level-0 CT (uncompressed 128^3
   chunks) and of compressed surface predictions by direct chunk HTTP. The
   fsspec-over-HTTP path times out under load; direct chunk requests with
   retries do not.
2. **Renderers without meshes.** The organizers publish surface predictions for
   every eligible scroll. A height-field renderer grows one field per sheet from
   the prediction's per-column peaks (after rotating a block so sheets are near
   vertical), smooths it, and samples 21 or 62 layers along the normal. Four
   variants cover 9 um blocks, 2.4 um blocks with 9.6 um surface predictions,
   large axis-aligned boxes, and tilted sheets (rotate only the small surface
   prediction, sample the original volume through inverse-rotated coordinates).
3. **Renderer for fitted meshes.** `render_tifxyz.py` takes a `tifxyz` winding
   from the spiral fitter (grid at 1/20 resolution), upsamples the coordinates,
   estimates normals, computes which 128^3 chunks the layers touch, fetches only
   those (cached), and samples. A 28 mm x 14 mm sector needs about 360 chunks
   (0.7 GB) and three minutes.
4. **Model drivers.** Flat 9 um model through the villa package; the canonical
   2 um ResNet3D-152 model through the villa `optimized_inference` code on a
   local numpy stack (no S3, no container); the full-3D DINO-guided U-Net with an
   in-memory sliding window (the generic runner writes ~30 GB of per-patch logits
   per 1024^3 block); dense DINO ink-likeness with the `dinovol` teacher.
5. **Sweeps.** `sweep.py` picks blocks on the outer windings of a scroll at
   several heights and runs the whole chain unattended, deleting raw blocks and
   keeping previews and scores. `batch_windings.py` does the same for every
   winding of a fitted mesh over a z-band.
6. **WSL2 recipe for the official spiral fitter.** `uv sync` with cmake and
   ninja provided by `uv tool install` (no sudo), CUDA torch, a `spiral-scroll.json`
   template, a lasagna slab copier that materialises only the z-band needed,
   and a one-line headless fit. Two config traps documented: `input_use_tracks`
   is off by default, and enabling tracks pulls in an `outer_shell` input that
   must be disabled unless present.

### 2.1 Relation to prior work

The Windows and consumer-GPU side of the spiral fitter was opened up in the July and August 2026 progress prizes
(Nicolas Dolegieviez, villa PR 1268: fit_spiral on Windows and loud failures for zeroed losses; Shuhan Yang,
spiral-fit-consumer-gpu: 12 GB cards, no native vc extension). This toolkit does not duplicate those patches: it runs
the stock fitter under WSL2 and adds what is needed after a fit exists. The new parts are the winding-to-ink-map
renderer that streams only the chunks a winding touches, unattended whole-mesh sweeps with a calibrated noise
ceiling, the per-scroll winding counts and estimated umbilici the fitter needs for scrolls without published ones,
and the systematic negative results in section 4.

## 3. Validation

| Test | Result |
|---|---|
| Flat model on a crop of Paris 4 segment 20231016151002 (level 2 of its 2.4 um surface volume, pooled to 21 slices) | Crisp Greek letters matching the official ink map; reverse direction is noise (figures `paris4_ctrl_flat.png`, `ink_ds8_crop.png`) |
| Canonical 2 um model through the height-field renderer on a raw Paris 4 block at a coordinate from the model's training config | 40% of one sheet above 0.5 (`sheet3_x515_canon.png`) |
| DINO ink-likeness on the same block | mean cosine to the reference ink embedding 0.485, sheet structure visible |

## 4. Findings on eligible scrolls

PHerc1203, 2.4 um scan (the only eligible scroll with one):

| Detector | Area | Result |
|---|---|---|
| Flat 9 um model, 21 published auto-grown segments of PHerc0800 and PHerc1447 | ~120 cm^2 | diffuse blobs, no rows; renders show sheet switching |
| Flat 9 um model, height-field sheets from the 9.36 um scan, 5 blocks | ~60 cm^2 | flat maps |
| Canonical 2 um model, four cleanly rendered sheets from a 2048^3 block on regular windings | ~4 x 0.4 cm^2 | forward: max 0.77, <1% above 0.5; reverse: 3-4% only along mask edges |
| Full-3D DINO-guided U-Net on two 1024^3 blocks | | amorphous clouds with patch seams; the same on a Paris 4 control, so not used |
| DINO ink-likeness, 2048^3 block | | mean cosine 0.393 vs 0.485 on Paris 4; 2.3% of papyrus tokens above z=2 vs 1.7%, but in fewer, larger clusters following sheets |

PHerc0125, 9.36 um scan: spiral fits over z 9000-10500 (300, 3000, 10000
steps; CW and ACW indistinguishable at 3000 steps; track satisfaction 12-15%,
which the fitter's own notes describe as normal). A sweep of the outer 20
windings through the flat model over a 4.7 mm band found no letter-like
structure (see `batch.log` in the results directory). Radial sheet counts
give roughly 60-90 windings for this scroll, so the Scroll 1 default of 130
was too high. A 30,000-step fit with 90 windings (track-radius loss 123 vs 182 at 10,000 steps with 130) renders clean continuous sheets, e.g. winding w089 over rows 20-60 (`pherc0125_w089_render.png`, 8 mm x 41 mm); its ink map is empty (`pherc0125_w089_ink.png`). The sweep of every third winding of that mesh (24 windings, 7.5 mm band, full circumference) is in `results/p0125_batch_acw30k`: all negative, max frac>200 0.0006. A second 1.5 mm z-band 3.5 mm higher (slices 12500-14000) was fitted and swept the same way (`results/p0125_b2_batch`, 23 windings): all negative, max frac>200 0.0009; the innermost winding w021 gives a diffuse blotchy map with no letter forms (`w021_ink.png`). The PHerc0826 second band (slices 6500-8000, `results/p0826_b2_batch`, 17 windings) is likewise negative; its highest winding w033 (robust score 7 blobs, frac>200 0.0038, just under the speckle ceiling) is a blotchy map without letter forms (`w033_ink.png`). The PHerc0211 second band (slices 13500-15000, `results/p0211_b2_batch`, 23 windings) is negative as well (max frac>200 0.0019 at w045). Two further scrolls were fitted from estimated umbilici: PHerc0358 (slices 8000-9500, 100 windings, 50.4% track points satisfied) and PHerc0257 (slices 10000-11500, 90 windings, 51.7%); their sweeps follow.

The negatives are consistent with the organizers' statement that ink remains
elusive in the new scans with current models. What the toolkit adds is that a
single-GPU contributor can now falsify or confirm that for any block of any
eligible scroll in minutes, and sweep unattended.


### 4.1 Sweep scoreboards (flat 9 um model, both layer directions)

Score = letter-sized bright blobs (400-30000 px above 200/255) and the fraction of pixels above 200.
For calibration, the Paris 4 control crop with clear letters has frac>200 = 0.0112; speckle noise gives up to ~0.004.

| Target | Units | Best blobs | Max frac>200 |
|---|---|---|---|
| PHerc1203, 9.36 um, height-field sheets, 4 outer-winding blocks | 32 renders | 2 | 0.0023 |
| PHerc0125, 10k-step spiral mesh, windings w091-w129 (every 2nd), rows 25-50 (4.7 mm band) | 40 renders | 17 (speckle) | 0.0044 |
| PHerc0268, 8.64 um, height-field sheets, 6 outer-winding blocks at two heights | 48 renders | 1 | 0.0008 |
| PHerc0211, 30k-step spiral mesh (90 windings, 17.5% tracks), windings w021-w087 (every 3rd), rows 20-60 (7.5 mm band) | 46 renders | 3 | 0.0016 |
| PHerc0826, 30k-step spiral mesh (70 windings, 12.6% tracks), windings w021-w069 (every 3rd), rows 20-60 (7.5 mm band) | 34 renders | 5 | 0.0017 |
| PHerc0211, 10k-step spiral mesh (100 windings, 12.9% tracks), windings w021-w099 (every 3rd), rows 20-60 (7.5 mm band) | 54 renders | 8 | 0.0030 |
| PHerc0826, 10k-step spiral mesh (100 windings, 12.9% tracks), windings w021-w099 (every 3rd), rows 20-60 (7.5 mm band) | 54 renders | 38 (w090, garbled render, see caveat); next 8 | 0.0114 (w090) / 0.0021 |
| PHerc0125, 30k-step spiral mesh (90 windings), windings w020-w089 (every 3rd), rows 20-60 (7.5 mm band, full circumference) | 48 renders | 3 | 0.0006 |
| PHerc0125, 30k-step spiral mesh (90 windings), second z-band (slices 12500-14000), windings w021-w087 (every 3rd), rows 20-60 (7.5 mm band) | 46 renders | 1 | 0.0009 |
| PHerc0826, 30k-step spiral mesh (70 windings), second z-band (slices 6500-8000), windings w021-w069 (every 3rd), rows 20-60 (7.5 mm band) | 34 renders | 4 | 0.0026 |
| PHerc0211, 30k-step spiral mesh (90 windings), second z-band (slices 13500-15000), windings w021-w087 (every 3rd), rows 20-60 (7.5 mm band) | 46 renders | 4 | 0.0019 |
| PHerc0800 + PHerc1447, 21 published auto-grown segments | 42 renders | 2 | 0.0197 (edge artefacts) |

All negative. Previews (`*_ink.png`) and per-render JSON scores are kept for every unit.

Caveat on the blob score: it is triage only. Where a fitted winding leaves the papyrus or skims a sheet edge-on, the flat model produces coarse
high-contrast clusters that score as "letter-sized blobs" (e.g. PHerc0826 winding w090: 38 blobs, frac>200 0.011, visually a garbled render).
`tools/rescore.py` removes blobs on mask boundaries and untextured fill, but garbled-yet-textured renders still pass, so every hit above the
speckle ceiling was inspected by eye against its render; none showed strokes or rows.


## 5. Reproduce

See the top-level README for the environment. The exact commands used for
every figure are in `tools/README.md`; the sweeps are `tools/sweep.py` and
`tools/batch_windings.py`; the WSL fit is `wsl/run_fit.sh <dataset> <out> 9000
10500 30000 ACW ', "input_use_tracks": true, "input_use_outer_shell": false,
"shell_outer_winding_idx": 90, "model_gap_expander_num_windings": 90'`.
