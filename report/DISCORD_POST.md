# Optional Discord announcement (paste yourself, e.g. in #progress-prizes or the tooling channel)

Hi all, I'm sharing vc-windows-tools, a single-GPU, Windows-native ink search toolkit:
https://github.com/gmDevi/vc-windows-tools

- Renders 21/62-layer sheet stacks straight from the public S3 zarrs (only the chunks a sheet touches, no local volume),
  from height fields, tilted boxes, or tifxyz windings of a fitted spiral mesh.
- Runs the published ink models (flat 9 um, canonical 2 um, 3D DINO-guided, DINO ink-likeness), scores the maps,
  and sweeps a whole fitted mesh unattended.
- Headless WSL2 recipe for the official spiral fitter, plus two config traps that silently give empty meshes
  (tracks off by default; outer-shell input pulled in when tracks are enabled).
- Winding-count estimates for the eight track-ready scrolls and estimated umbilici for six that have none published.
- Results: ~350 rendered sheet units across PHerc1203, 0125, 0826, 0211, 0268, 0800 and 1447, all negative, with a
  calibrated speckle ceiling (frac>200 about 0.004 vs 0.011 for real letters on Paris 4). Every run's scores and PNGs
  are in results/. Feedback and pull requests welcome.
