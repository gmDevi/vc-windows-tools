"""Render every winding tifxyz of a fitted spiral mesh directory for a row band, run the flat ink model both ways, score, keep PNGs.
Usage: batch_windings.py <meshes_fitted_dir> <zarr_rel_path/> <out_root> [row0 row1] [ckpt]"""
import sys, os, json, glob, subprocess, shutil, time
R = 'C:/Users/mdevi/prize/vesuvius'; VENV = R + '/villa/vesuvius'
mesh_dir, zrel, out_root = sys.argv[1:4]
row0, row1 = (sys.argv[4], sys.argv[5]) if len(sys.argv) > 5 else (None, None)
ckpt = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != '-' else R + '/checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth'
wsel = sys.argv[7] if len(sys.argv) > 7 else None   # e.g. '129:89:-2' -> windings w129 down to w090 every 2nd
os.makedirs(out_root, exist_ok=True)
log = open(os.path.join(out_root, 'batch.log'), 'a')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
def uvrun(args):
    return subprocess.run(['uv', 'run', '--no-sync', 'python'] + args, cwd=VENV, capture_output=True, text=True)
windings = sorted(d for d in glob.glob(os.path.join(mesh_dir, 'w[0-9][0-9][0-9]')) if os.path.isdir(d))
if wsel:
    a, b, st = (int(v) for v in wsel.split(':'))
    keep = {f'w{i:03d}' for i in range(a, b, st)}
    windings = [d for d in windings if os.path.basename(d) in keep]
    windings.sort(key=lambda d: -int(os.path.basename(d)[1:]) if st < 0 else int(os.path.basename(d)[1:]))
say('windings', len(windings))
results = []
for wd in windings:
    name = os.path.basename(wd); z = os.path.join(out_root, name + '.zarr')
    if os.path.exists(os.path.join(out_root, name + '_summary.json')): continue
    t0 = time.time()
    args = [R + '/tools/render_tifxyz.py', wd, zrel, z, '21']
    if row0 is not None: args += ['0', '100000', row0, row1]
    r = uvrun(args)
    if 'wrote' not in r.stdout: say(name, 'render failed', r.stderr[-300:]); continue
    tif = os.path.join(out_root, name + '_ink.tif')
    uvrun(['-m', 'vesuvius.ink_detection.inference.infer', z, ckpt, tif, '--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '4', '--direction', 'both', '--no-compile'])
    summ = {}
    for suf in ['', '_reverse']:
        t = os.path.join(out_root, name + '_ink' + suf + '.tif')
        if not os.path.exists(t): continue
        uvrun([R + '/tools/tif2png.py', t, t[:-4] + '.png', '2'])
        s = uvrun([R + '/tools/score_tif.py', t])
        try: summ['ink' + suf] = json.loads(s.stdout.strip().splitlines()[-1])
        except Exception: summ['ink' + suf] = {'error': s.stderr[-200:]}
        os.remove(t)
    shutil.rmtree(z, ignore_errors=True)
    json.dump(summ, open(os.path.join(out_root, name + '_summary.json'), 'w'), indent=1)
    best = max((v.get('blobs_letterlike', 0) for v in summ.values()), default=0)
    say(f'{name}: best letter-like blobs {best}, frac200 {max((v.get("frac200", 0) for v in summ.values()), default=0):.5f}, {time.time()-t0:.0f}s')
    results.append((name, best))
say('DONE', sorted(results, key=lambda r: -r[1])[:8])
