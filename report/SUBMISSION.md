# September 2026 Progress Prize form: prepared answers

Form: https://docs.google.com/forms/d/e/1FAIpQLScNBMj25FMnphngRG1Ciryv_2_Mkdq2YPJOD9WqPfZExII2iQ/viewform

**Email.** mdevincenzis@gmail.com

**Your full name.** Marco De Vincenzi

**Team description.** Individual submission (Marco De Vincenzi), with an AI
coding assistant (Claude) used for implementation.

**Discord display name.** hamiltonian

**URL of your open source contribution.** https://github.com/gmDevi/vc-windows-tools
(MIT license; commits, results, and figures included) and villa pull request
https://github.com/ScrollPrize/villa/pull/1732 (fit_spiral: fit from tracks
when the dataset has no outer shell)

**What is your contribution?**

(1) Scroll data: the 2025-2026 eligible scrolls, mainly PHerc1203 (2.4 um and
9.36 um scans), PHerc0125, PHerc0826 and PHerc0211 (9.36 um, fitted spiral
meshes), PHerc0268 (8.64 um), plus the published auto-grown segments of
PHerc0800 and PHerc1447; Paris 4 was used as the
positive control.

(2) How it raises the probability of reading these scrolls: it removes the
infrastructure barrier for single-GPU contributors. From the organizers'
public CT volumes and surface predictions, with no local copy of any volume,
the toolkit fetches only the chunks a sheet touches, renders 21- or 62-layer
face-on sheet stacks (four renderers: height fields grown from the surface
predictions at 9 um and 2.4 um, big axis-aligned boxes, tilted sheets without
rotating the volume, and tifxyz windings from a fitted spiral), runs the
published ink models (flat 9 um, canonical 2 um, full-3D DINO-guided, DINO
ink-likeness), scores the maps, and sweeps a whole scroll or a whole fitted
mesh unattended. It also makes the official spiral fitter usable under WSL2
on Windows with a headless one-line recipe, documents two config traps that
silently produce empty meshes (tracks off by default; the outer-shell input
pulled in when tracks are enabled), and ships a renderer that turns fitted
windings into ink maps from streamed S3 chunks in minutes. The fitter side
builds on the July/August Windows and consumer-GPU work by Nicolas
Dolegieviez (villa PR 1268) and Shuhan Yang; what is new here is the
downstream half (fitted winding to ink map with no local volume, whole-mesh
sweeps with a calibrated noise ceiling) and the per-scroll inputs the fitter
needs but the data does not ship: winding counts for eight track-ready
scrolls and estimated umbilici for six of them. More people can
therefore test more models on more sheets of more scrolls, which is how the
first letters in a new scroll will be found.

(3) What it enables that was not possible before: a Windows machine with one
consumer GPU and under 100 GB of free disk can go from "pick a scroll and a
location" to an ink map in minutes, sweep hundreds of square centimetres
overnight, and run the official spiral fitter, none of which the current
tooling supports without Linux, a C++ toolchain, and terabytes of storage.
It also provides winding-count estimates for the eight track-ready scrolls
(the fitter's Scroll 1 default of 130 is wrong for them; PHerc0125 has about
60, PHerc0826 about 55).

(4) Evidence: the flat model through the toolkit reproduces the organizers'
ink map of Paris 4 segment 20231016151002 (figure in the repo); the canonical
2 um model through the height-field renderer fires on 40% of a Paris 4 sheet
at a coordinate from its own training config; the fitted PHerc0125 mesh
renders clean continuous sheets (figure). Every sweep run is logged with
per-render JSON scores and PNG previews in `results/`: PHerc1203 (2.4 um,
three detectors, ~1 cm^2 of clean sheets), PHerc1203 9 um blocks, PHerc0125
(three spiral meshes over two z-bands, 67 windings), PHerc0826 (two meshes,
44 windings), PHerc0211 (two meshes, 50 windings), PHerc0268 (6 blocks), and
the 21 published PHerc0800/PHerc1447 segments. All negative, which is itself
useful: it is the first systematic single-machine falsification of the
current models on these scrolls, with a calibrated noise ceiling (frac>200 of
0.004 for speckle vs 0.011 for real letters).

**Terms and Conditions.** Yes, I agree.
