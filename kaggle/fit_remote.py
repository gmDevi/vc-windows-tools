"""Kaggle kernel: fit one z-band of one scroll with the official spiral fitter (tracks only) and save the fitted
tifxyz windings as the kernel output. Edit JOB below, push with `kaggle kernels push -p kaggle/`, fetch results with
`kaggle kernels output gmdevi/<slug> -p <dir>`. Runs on a T4/P100 (needs ~7 GB VRAM); 30k steps take a few hours."""
import os, subprocess, sys, json, time, zipfile, urllib.request, shutil, glob
JOB = dict(
    scroll="PHerc0191", volume="20250821151635", lasagna_run="20260419180421",
    z0=9000, z1=10500, sense="CW", nwind=100, steps=30000,
    umbilicus_url="https://raw.githubusercontent.com/gmDevi/vc-windows-tools/master/results/umbilici/PHerc0191_est.json",
    tag="cw30k_w100",
)
if os.environ.get("FIT_JOB"): JOB.update(json.loads(os.environ["FIT_JOB"]))      # Colab / any pod: override the job from the environment
HOME = os.path.expanduser("~"); WORK = os.environ.get("FIT_WORK", "/kaggle/working"); TMP = os.environ.get("FIT_TMP", "/kaggle/tmp"); os.makedirs(TMP, exist_ok=True); os.makedirs(WORK, exist_ok=True)
def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True); t = time.time()
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", **kw)
    print(f"  -> rc={r.returncode} in {time.time()-t:.0f}s", flush=True)
    if r.returncode: raise SystemExit(f"failed: {cmd}")
def get(url, dst, tries=6):
    for i in range(tries):
        try:
            sh(f"curl -sSL --retry 5 --retry-all-errors -C - -o '{dst}' '{url}'"); return
        except SystemExit:
            time.sleep(10)
    raise SystemExit(f"download failed: {url}")

# 0. internet check (Kaggle only enables it for phone-verified accounts)
sh("curl -fsS -m 20 https://github.com -o /dev/null")
# 1. toolchain: uv + Python 3.14 + spiral-fitting (native helpers build with the image's gcc)
sh("curl -fsSL https://astral.sh/uv/install.sh -o /tmp/uv.sh && sh /tmp/uv.sh")  # no pipe: a failed download must fail the step
os.environ["PATH"] = f"{HOME}/.local/bin:" + os.environ["PATH"]
if not os.path.isdir(f"{TMP}/villa"):
    sh(f"git clone --depth 1 https://github.com/ScrollPrize/villa.git {TMP}/villa")
sh(f"cd {TMP}/villa/spiral-fitting && uv python install 3.14 && uv sync")
sh(f"cd {TMP}/villa/spiral-fitting && uv run --no-sync python -c \"import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))\"")

# 2. dataset: tracks (+ crossings sidecar, verified as a zip), umbilicus, spiral-scroll.json, lasagna slab
S, V = JOB["scroll"], JOB["volume"]; D = f"{TMP}/spiral_data/{S}"; os.makedirs(f"{D}/tracks", exist_ok=True); os.makedirs(f"{D}/lasagna_inputs", exist_ok=True)
B = f"https://dl.ash2txt.org/datasets/spiral_datasets/{S}/{V}/tracks"; N = f"{S}_{V}_surface_m7_L0_th0.2"
get(f"{B}/{N}.extract.json", f"{D}/tracks/{N}.extract.json")
get(f"{B}/{N}.dbm.crossings.npz", f"{D}/tracks/{N}.dbm.crossings.npz")
if not zipfile.is_zipfile(f"{D}/tracks/{N}.dbm.crossings.npz"): raise SystemExit("crossings.npz is not a valid zip (truncated download)")
get(f"{B}/{N}.dbm", f"{D}/tracks/{N}.dbm")
get(JOB["umbilicus_url"], f"{D}/umbilicus.json")
json.dump({"schema_version": 1, "name": S, "voxel_size_um": 9.362, "spiral_outward_sense": JOB["sense"], "normal_zarr_group": "2",
           "lasagna_scale": 4, "paths": {"tracks_dbm": f"tracks/{N}.dbm"}}, open(f"{D}/spiral-scroll.json", "w"), indent=1)
get("https://raw.githubusercontent.com/gmDevi/vc-windows-tools/master/wsl/fetch_lasagna_slab2.py", f"{TMP}/fetch_lasagna_slab2.py")
z0, z1 = JOB["z0"], JOB["z1"]
for attempt in range(6):  # the slab copier resumes; retry on dropped connections
    r = subprocess.run(f"cd {TMP} && uv run --python 3.12 --with 'zarr>=2.18,<3' --with fsspec --with aiohttp --with requests --with numcodecs --with numpy "
                       f"python fetch_lasagna_slab2.py {S} {V} {JOB['lasagna_run']} 2 {z0//4-50} {z1//4+50} {D}/lasagna_inputs", shell=True, executable="/bin/bash")
    print(f"lasagna fetch attempt {attempt+1}: rc={r.returncode}", flush=True)
    if r.returncode == 0: break
    time.sleep(30)
else:
    raise SystemExit("lasagna fetch failed after 6 attempts")
sh(f"du -sh {D}/tracks {D}/lasagna_inputs; df -h {TMP} {WORK}")

# 3. fit (headless CLI; overrides as JSON in the environment, see spiral-fitting/README.md)
OUT = f"{TMP}/spiral_out/{S}_{JOB['tag']}"; os.makedirs(OUT, exist_ok=True)
overrides = {"z_begin": z0, "z_end": z1, "optimizer_num_training_steps": JOB["steps"], "input_disable_patches": True,
             "loss_weight_shell_outer": 0, "loss_weight_shell_patch_radius": 0, "dense_spacing_mode": "grad_mag",
             "loss_weight_dense_spacing": 0, "input_use_tracks": True, "input_use_outer_shell": False,
             "shell_outer_winding_idx": JOB["nwind"], "model_gap_expander_num_windings": JOB["nwind"]}
env = dict(os.environ, FIT_SPIRAL_OUT_DIR=OUT, FIT_SPIRAL_CONFIG_OVERRIDES=json.dumps(overrides), WANDB_MODE="disabled")
print("overrides:", json.dumps(overrides), flush=True)
r = subprocess.run(f"cd {TMP}/villa/spiral-fitting && uv run --no-sync python fit_spiral.py --dataset {D} --cache {TMP}/spiral_cache 2>&1 | tee {OUT}/fit.log",
                   shell=True, executable="/bin/bash", env=env)

# 4. output: fitted windings + log (small), regardless of exit code so partial runs can be inspected
dst = f"{WORK}/meshes_{S}_{JOB['tag']}"; os.makedirs(dst, exist_ok=True)
for w in sorted(glob.glob(f"{OUT}/*/meshes/fitted/w[0-9][0-9][0-9]")): shutil.copytree(w, f"{dst}/{os.path.basename(w)}", dirs_exist_ok=True)
for f in glob.glob(f"{OUT}/*/satisf*.json") + [f"{OUT}/fit.log"]:
    if os.path.exists(f): shutil.copy(f, dst)
json.dump(JOB, open(f"{dst}/job.json", "w"), indent=1)
sh(f"du -sh {dst}; ls {dst} | head; tail -n 5 {OUT}/fit.log")
print("FIT_EXIT", r.returncode, flush=True)
