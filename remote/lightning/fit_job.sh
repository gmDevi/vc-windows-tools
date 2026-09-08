#!/usr/bin/env bash
# Runs INSIDE a Lightning studio (Ubuntu, GPU). One spiral fit of one z-band; survives our disconnect (started with nohup).
# Inputs via env: SCROLL VOLUME LASAGNA_RUN Z0 Z1 SENSE NWIND STEPS UMB_URL TAG
# Layout: ~/vesuvius/{villa,spiral_data/<scroll>,spiral_out/<scroll>_<tag>,jobs/<scroll>_<tag>.{log,done}}
set -uo pipefail
ROOT=/teamspace/studios/this_studio/vesuvius; J=$ROOT/jobs; mkdir -p $J $ROOT/spiral_out; JOB=${SCROLL}_${TAG}; LOG=$J/$JOB.log; exec > >(tee -a "$LOG") 2>&1
echo "=== fit_job start $(date -u +%FT%TZ) $JOB"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export PATH="$HOME/.local/bin:$PATH"; export DEBIAN_FRONTEND=noninteractive
step() { echo "--- $1 $(date -u +%T)"; }
step toolchain
command -v uv >/dev/null || (curl -fsSL https://astral.sh/uv/install.sh -o /tmp/uv.sh && sh /tmp/uv.sh >/dev/null)
[ -d $ROOT/villa ] || git clone --depth 1 https://github.com/ScrollPrize/villa.git $ROOT/villa
cd $ROOT/villa/spiral-fitting && (uv python install 3.14 && uv sync) >/dev/null 2>&1 || { echo "uv sync failed"; echo FAILED > $J/$JOB.done; exit 1; }
uv run --no-sync python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))" || { echo FAILED > $J/$JOB.done; exit 1; }
step dataset
D=$ROOT/spiral_data/$SCROLL; mkdir -p $D/tracks $D/lasagna_inputs; B="https://dl.ash2txt.org/datasets/spiral_datasets/$SCROLL/$VOLUME/tracks"; N="${SCROLL}_${VOLUME}_surface_m7_L0_th0.2"
get() { for i in 1 2 3 4 5; do curl -sSL --retry 5 --retry-all-errors -C - -o "$2" "$1" && return 0; sleep 10; done; return 1; }
get "$B/$N.extract.json" $D/tracks/$N.extract.json
[ -f $D/tracks/$N.dbm.crossings.npz ] && python3 -c "import zipfile,sys;sys.exit(0 if zipfile.is_zipfile('$D/tracks/$N.dbm.crossings.npz') else 1)" || get "$B/$N.dbm.crossings.npz" $D/tracks/$N.dbm.crossings.npz
python3 -c "import zipfile,sys;sys.exit(0 if zipfile.is_zipfile('$D/tracks/$N.dbm.crossings.npz') else 1)" || { echo "crossings.npz invalid"; echo FAILED > $J/$JOB.done; exit 1; }
get "$B/$N.dbm" $D/tracks/$N.dbm
get "$UMB_URL" $D/umbilicus.json
printf '{"schema_version":1,"name":"%s","voxel_size_um":9.362,"spiral_outward_sense":"%s","normal_zarr_group":"2","lasagna_scale":4,"paths":{"tracks_dbm":"tracks/%s.dbm"}}\n' $SCROLL $SENSE $N > $D/spiral-scroll.json
get https://raw.githubusercontent.com/gmDevi/vc-windows-tools/master/wsl/fetch_lasagna_slab2.py $ROOT/fetch_lasagna_slab2.py
ok=0; for attempt in 1 2 3 4 5 6; do cd $ROOT && uv run --python 3.12 --with 'zarr>=2.18,<3' --with fsspec --with aiohttp --with requests --with numcodecs --with numpy python fetch_lasagna_slab2.py $SCROLL $VOLUME $LASAGNA_RUN 2 $((Z0/4-50)) $((Z1/4+50)) $D/lasagna_inputs && { ok=1; break; }; echo "lasagna fetch attempt $attempt failed; retrying"; sleep 30; done
[ $ok = 1 ] || { echo "lasagna fetch failed after 6 attempts"; echo FAILED > $J/$JOB.done; exit 1; }
du -sh $D/tracks $D/lasagna_inputs; df -h $HOME | tail -1
step fit
OUT=$ROOT/spiral_out/$JOB; mkdir -p $OUT
export FIT_SPIRAL_OUT_DIR=$OUT WANDB_MODE=disabled
export FIT_SPIRAL_CONFIG_OVERRIDES="{\"z_begin\": $Z0, \"z_end\": $Z1, \"optimizer_num_training_steps\": $STEPS, \"input_disable_patches\": true, \"loss_weight_shell_outer\": 0, \"loss_weight_shell_patch_radius\": 0, \"dense_spacing_mode\": \"grad_mag\", \"loss_weight_dense_spacing\": 0, \"input_use_tracks\": true, \"input_use_outer_shell\": false, \"shell_outer_winding_idx\": $NWIND, \"model_gap_expander_num_windings\": $NWIND}"
cd $ROOT/villa/spiral-fitting && uv run --no-sync python fit_spiral.py --dataset $D --cache $ROOT/spiral_cache 2>&1 | grep --line-buffered -v "^PROGRESS Optimizing" | tee $OUT/fit.log | tail -n 3
RC=${PIPESTATUS[0]}
step export
DST=$ROOT/jobs/meshes_$JOB; rm -rf $DST; mkdir -p $DST
for w in $OUT/*/meshes/fitted/w[0-9][0-9][0-9]; do [ -d "$w" ] && cp -r "$w" $DST/; done
cp $OUT/*/satisf*.json $DST/ 2>/dev/null; cp $OUT/fit.log $DST/
(cd $ROOT/jobs && tar czf meshes_$JOB.tgz meshes_$JOB) && du -sh $ROOT/jobs/meshes_$JOB.tgz
echo "fit rc=$RC windings=$(ls -d $DST/w* 2>/dev/null | wc -l)"
[ "$RC" = 0 ] && echo DONE > $J/$JOB.done || echo FAILED > $J/$JOB.done
echo "=== fit_job end $(date -u +%FT%TZ)"
