# September 2026 Progress Prize form: prepared answers

Form: https://docs.google.com/forms/d/e/1FAIpQLScNBMj25FMnphngRG1Ciryv_2_Mkdq2YPJOD9WqPfZExII2iQ/viewform

**Email.** mdevincenzis@gmail.com

**Your full name.** Gennaro Marco Devincenzis

**Team description.** Individual submission (Gennaro Marco Devincenzis), with an AI
coding assistant (Claude) used for implementation.

**Discord display name.** hamiltonian

**URL of your open source contribution.** https://github.com/gmDevi/vc-windows-tools
(MIT license; commits, results, and figures included) and villa pull request
https://github.com/ScrollPrize/villa/pull/1732 (fit_spiral: fit from tracks
when the dataset has no outer shell) and
https://github.com/ScrollPrize/villa/pull/1735 (fit_spiral: a truncated
crossings cache no longer aborts the fit) and
https://github.com/ScrollPrize/villa/pull/1736 (README recipe for tracks-only
scrolls, umbilicus and winding-count estimators). Browsable results and an
in-browser scorer: https://huggingface.co/spaces/gmDevi/vesuvius-ink-sweeps

**What is your contribution?**

(1) Scroll data: the 2025-2026 eligible scrolls, mainly PHerc1203 (2.4 um and
9.36 um scans), PHerc0125, PHerc0826 and PHerc0211 (9.36 um, fitted spiral
meshes), PHerc0268 (8.64 um), plus the published auto-grown segments of
PHerc0800 and PHerc1447; Paris 4 was used as the
positive control.

(2) What is new, in order of importance. First, a renderer that turns the
spiral fitter's tifxyz windings into face-on ink maps directly from streamed
S3 chunks, with no local copy of any volume, in minutes per winding, and a
batch driver that sweeps every winding of a fitted mesh unattended with a
per-render JSON score and PNG preview. Second, a calibrated noise ceiling for
the flat 9 um model (frac>200 of 0.004 for speckle versus 0.011 for real
letters, measured on the Paris 4 control and on crushed regions) that turns
"nothing found" into a quantified negative, and the first systematic
single-machine falsification of the current models on the eligible scrolls:
PHerc1203 (2.4 um, three detectors), PHerc0125 (three meshes, 67 windings),
PHerc0826 (44 windings), PHerc0211 (50 windings), PHerc0358 (27 windings,
where high blob counts trace to a crushed region), PHerc0268 (6 blocks), the
21 published PHerc0800/PHerc1447 segments, and PHerc0813 (46 renders, max
frac>200 0.0013), PHerc0191 (54 renders, 0.0011) and PHerc0257 (46 renders,
0.0023), the last three fitted and swept on free Kaggle T4 GPUs. Third,
the per-scroll inputs the fitter needs but the data does not ship: winding
counts for eight track-ready scrolls (the fitter's Scroll 1 default of 130 is
wrong for them; PHerc0125 has about 60, PHerc0826 about 55) and estimated
umbilici for six of them, validated by the fraction of satisfied track points
(PHerc0358 50.4%, PHerc0257 51.7%), plus two upstream fixes for silent
empty-mesh traps (villa PRs 1732 and 1735) and the README recipe (PR 1736).
Fourth, the whole chain runs on Windows with WSL2 and one consumer GPU under
100 GB of disk; that part builds on the July/August Windows and consumer-GPU
work by Nicolas Dolegieviez (villa PR 1268) and Shuhan Yang.

(3) How it raises the probability of reading these scrolls: more people can
test more models on more sheets of more scrolls. From the organizers' public
CT volumes and surface predictions the toolkit fetches only the chunks a
sheet touches, renders 21- or 62-layer sheet stacks with four renderers
(height fields grown from the surface predictions at 9 um and 2.4 um, big
axis-aligned boxes, tilted sheets without rotating the volume, and fitted
spiral windings), runs the published ink models (flat 9 um, canonical 2 um,
full-3D DINO-guided, DINO ink-likeness), scores the maps and sweeps a whole
scroll or a whole mesh overnight, none of which the current tooling supports
without Linux, a C++ toolchain and terabytes of storage. Negative results
with a stated threshold are published so that nobody repeats them, and the
same harness runs unattended on free Kaggle GPUs (recipe in the repo).

(4) Evidence: the flat model through the toolkit reproduces the organizers'
ink map of Paris 4 segment 20231016151002 (figure in the repo); the canonical
2 um model through the height-field renderer fires on 40% of a Paris 4 sheet
at a coordinate from its own training config; the fitted PHerc0125 mesh
renders clean continuous sheets (figure). Every sweep run is logged with
per-render JSON scores and PNG previews in `results/` (about 400 scored
renders), with the scoreboards in `report/REPORT.md`.

**Terms and Conditions.** Yes, I agree.
