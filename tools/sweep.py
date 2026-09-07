"""Unattended sweep over one scroll: pick blocks on the outer windings at several heights, fetch CT + surface-prediction blocks,
render sheets (heightfield_render.py), run the flat ink model both directions, score, keep PNG previews + JSON, delete the blocks.
Usage: sweep.py <scroll_id> <ct_zarr_rel_path/> <surf_zarr_rel_path/> <out_root> [n_z=4] [per_z=4] [block=1024]
"""
import sys, os, json, subprocess, shutil, time, urllib.request, urllib.parse, numpy as np, concurrent.futures as cf
from scipy import ndimage as ndi
from PIL import Image

R = 'C:/Users/mdevi/prize/vesuvius'; VENV = R + '/villa/vesuvius'
B = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
scroll, ct_rel, surf_rel, out_root = sys.argv[1:5]
n_z = int(sys.argv[5]) if len(sys.argv) > 5 else 4
per_z = int(sys.argv[6]) if len(sys.argv) > 6 else 4
N = int(sys.argv[7]) if len(sys.argv) > 7 else 1024
CKPT = R + '/checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth'
os.makedirs(out_root, exist_ok=True)
log = open(os.path.join(out_root, 'sweep.log'), 'a')

def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

def uvrun(args, **kw):
    return subprocess.run(['uv', 'run', '--no-sync', 'python'] + args, cwd=VENV, capture_output=True, text=True, **kw)

za = json.loads(urllib.request.urlopen(B + urllib.parse.quote(ct_rel + '0/.zarray')).read().decode()); shape0 = za['shape']
lvl = '4'; f = 16
za4 = json.loads(urllib.request.urlopen(B + urllib.parse.quote(ct_rel + lvl + '/.zarray')).read().decode()); sh = za4['shape']; c = za4['chunks'][0]

def read_lvl4_slab(zi):
    vol = np.zeros((c, sh[1], sh[2]), np.uint8)
    def get(iy, ix):
        try:
            return iy, ix, np.frombuffer(urllib.request.urlopen(B + urllib.parse.quote(f'{ct_rel}{lvl}/{zi}/{iy}/{ix}')).read(), np.uint8).reshape(c, c, c)
        except Exception:
            return iy, ix, None
    jobs = [(iy, ix) for iy in range((sh[1] + c - 1) // c) for ix in range((sh[2] + c - 1) // c)]
    with cf.ThreadPoolExecutor(16) as ex:
        for iy, ix, blk in ex.map(lambda p: get(*p), jobs):
            if blk is None: continue
            y1 = min(sh[1], (iy + 1) * c); x1 = min(sh[2], (ix + 1) * c)
            vol[:, iy * c:y1, ix * c:x1] = blk[:, :y1 - iy * c, :x1 - ix * c]
    return vol

# ---- choose blocks on the outer windings at several heights ----
blocks = []
for zc in np.linspace(0.2, 0.8, n_z) * shape0[0]:
    zi = int(zc) // f // c
    slab = read_lvl4_slab(zi); img = slab[c // 2].astype(np.float32)
    zmid0 = (zi * c + c // 2) * f
    mask = ndi.binary_opening(ndi.gaussian_filter(img, 2) > 25, iterations=2)
    lab, n = ndi.label(mask)
    if n == 0:
        say('no scroll mask at z', zmid0); continue
    sizes = ndi.sum(mask, lab, range(1, n + 1)); mask = ndi.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))
    inward = max(1, int(0.3 * N / f))                      # block centre sits ~0.3 block inside the outer boundary
    ring = ndi.binary_erosion(mask, iterations=inward) & ~ndi.binary_erosion(mask, iterations=inward + 3)
    ys, xs = np.where(ring)
    if ys.size == 0:
        say('empty ring at z', zmid0); continue
    cy, cx = ndi.center_of_mass(mask)
    ang = np.arctan2(ys - cy, xs - cx)
    for t in np.linspace(-np.pi, np.pi, per_z, endpoint=False) + np.pi / per_z:
        k = int(np.argmin(np.abs(((ang - t + np.pi) % (2 * np.pi)) - np.pi)))
        y0 = int(np.clip(ys[k] * f - N // 2, 0, shape0[1] - N)) // 128 * 128
        x0 = int(np.clip(xs[k] * f - N // 2, 0, shape0[2] - N)) // 128 * 128
        z0 = int(np.clip(zmid0 - N // 2, 0, shape0[0] - N)) // 128 * 128
        blocks.append((z0, y0, x0))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(out_root, f'lvl4_z{zmid0}.png'))
say(scroll, 'blocks:', blocks)

results = []
for (z0, y0, x0) in blocks:
    tag = f'{scroll}_z{z0}_y{y0}_x{x0}'; bdir = os.path.join(out_root, tag)
    if os.path.exists(os.path.join(bdir, 'summary.json')):
        say('skip', tag); continue
    os.makedirs(bdir, exist_ok=True); t0 = time.time()
    tmp = os.path.join(R, 'data', 'sweep_tmp'); os.makedirs(tmp, exist_ok=True)
    ct = os.path.join(tmp, tag + '_ct.zarr'); sf = os.path.join(tmp, tag + '_surf.zarr')
    subprocess.run([sys.executable, R + '/tools/fetch_block.py', ct_rel, str(z0), str(y0), str(x0), str(N), ct], capture_output=True, text=True)
    uvrun([R + '/tools/fetch_region_zarr.py', B + surf_rel + '0', str(z0), str(y0), str(x0), str(N), sf])
    if not (os.path.exists(os.path.join(ct, '0', '.zarray')) and os.path.exists(os.path.join(sf, '0', '.zarray'))):
        say('download failed', tag); continue
    r = uvrun([R + '/tools/heightfield_render.py', ct, sf, bdir, '4'])
    say(r.stdout[-500:].strip(), r.stderr[-300:].strip())
    summ = {'block': [z0, y0, x0], 'sheets': {}}
    for fn in sorted(os.listdir(bdir)):
        if not (fn.startswith('sheet') and fn.endswith('.zarr')): continue
        zpath = os.path.join(bdir, fn); n = fn[:-5]; tif = os.path.join(bdir, n + '_ink.tif')
        uvrun(['-m', 'vesuvius.ink_detection.inference.infer', zpath, CKPT, tif, '--overlap', '0.5', '--blend-mode', 'hann',
               '--batch-size', '4', '--direction', 'both', '--no-compile'])
        for suf in ['', '_reverse']:
            t = os.path.join(bdir, n + '_ink' + suf + '.tif')
            if not os.path.exists(t): continue
            uvrun([R + '/tools/tif2png.py', t, t[:-4] + '.png', '2'])
            r2 = uvrun([R + '/tools/score_tif.py', t])
            try:
                summ['sheets'][n + suf] = json.loads(r2.stdout.strip().splitlines()[-1])
            except Exception:
                summ['sheets'][n + suf] = {'error': r2.stderr[-300:]}
            os.remove(t)
        shutil.rmtree(zpath, ignore_errors=True)
    json.dump(summ, open(os.path.join(bdir, 'summary.json'), 'w'), indent=1)
    shutil.rmtree(ct, ignore_errors=True); shutil.rmtree(sf, ignore_errors=True)
    best = max((v.get('blobs_letterlike', 0) for v in summ['sheets'].values()), default=0)
    say(f'{tag}: {len(summ["sheets"])} renders, best letter-like blobs={best}, {time.time() - t0:.0f}s')
    results.append((tag, best))
say('DONE', scroll, sorted(results, key=lambda r: -r[1])[:5])
