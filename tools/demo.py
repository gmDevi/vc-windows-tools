"""One-command demo: scroll block -> sheet renders -> flat ink maps, from public data only.
Usage (from villa/vesuvius, inside its uv env):
  python tools/demo.py <scroll> <volume_id> <surface_pred_id> z0 y0 x0 [n=1024] [ckpt] [out_root]
Example:
  python tools/demo.py PHerc1203 20250820131727-9.362um-1.2m-113keV 20260413222639 8704 3072 1152 1024
Produces <out_root>/<scroll>_z<z0>_y<y0>_x<x0>/sheetN_*_ink.png (+ _reverse) and a summary.json with scores."""
import sys, os, json, subprocess, glob, time
here = os.path.dirname(os.path.abspath(__file__))
scroll, vol, pred, z0, y0, x0 = sys.argv[1:7]
n = sys.argv[7] if len(sys.argv) > 7 else '1024'
ckpt = sys.argv[8] if len(sys.argv) > 8 else os.path.join(here, '..', 'checkpoints', 'ink_9um', 'hybrid_3d2d-seed42', 'step-075000.pth')
out_root = sys.argv[9] if len(sys.argv) > 9 else os.path.join(here, '..', 'preds')
tag = f'{scroll}_z{z0}_y{y0}_x{x0}'; out = os.path.join(out_root, tag); os.makedirs(out, exist_ok=True)
ct_rel = f'{scroll}/volumes/{vol}-masked.zarr/'
surf_url = f'https://vesuvius-challenge-open-data.s3.amazonaws.com/{scroll}/representations/predictions/surfaces/{vol.split("-")[0]}-surface-{pred}-surface-m7-L0-th0.2.zarr/0'
ct = os.path.join(out, 'ct.zarr'); sf = os.path.join(out, 'surf')
py = sys.executable
def run(args):
    print('>', ' '.join(args[:3]), '...', flush=True); t = time.time()
    r = subprocess.run([py] + args, capture_output=True, text=True)
    print(r.stdout[-400:].strip(), r.stderr[-300:].strip() if r.returncode else '', f'({time.time()-t:.0f}s)', flush=True)
    return r.returncode == 0
if not os.path.exists(os.path.join(ct, '0', '.zarray')): run([os.path.join(here, 'fetch_block.py'), ct_rel, z0, y0, x0, n, ct])
if not os.path.exists(os.path.join(sf, '0', '.zarray')): run([os.path.join(here, 'fetch_region_zarr5.py'), surf_url, z0, y0, x0, n, n, n, sf])
run([os.path.join(here, 'heightfield_render.py'), ct, sf, out, '4'])
summary = {}
for z in sorted(glob.glob(os.path.join(out, 'sheet*.zarr'))):
    name = os.path.basename(z)[:-5]; tif = os.path.join(out, name + '_ink.tif')
    run(['-m', 'vesuvius.ink_detection.inference.infer', z, ckpt, tif, '--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '4', '--direction', 'both', '--no-compile'])
    for suf in ['', '_reverse']:
        t = os.path.join(out, name + '_ink' + suf + '.tif')
        if not os.path.exists(t): continue
        run([os.path.join(here, 'tif2png.py'), t, t[:-4] + '.png', '2'])
        r = subprocess.run([py, os.path.join(here, 'score_tif.py'), t], capture_output=True, text=True)
        try: summary[name + suf] = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception: summary[name + suf] = {'error': r.stderr[-200:]}
json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
