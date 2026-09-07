"""Render a surface volume from a tifxyz mesh (x/y/z.tif grids at meta 'scale', e.g. 0.05 = one grid cell per 20 voxels)
by upsampling the grid to native resolution, estimating normals, and sampling a public level-0 uncompressed zarr volume along
the normal (n_layers layers, 1 voxel apart). Chunks are fetched from S3 over HTTP on demand and cached locally.
Usage: render_tifxyz.py <tifxyz_dir> <zarr_rel_path/> <out.zarr> [n_layers=21] [col0 col1] [row0 row1] [cache_dir]
Output: zarr group with '0' of shape (n_layers, H, W) uint8, plus a mid-layer PNG."""
import sys, os, json, time, urllib.request, urllib.parse, numpy as np, tifffile, zarr, concurrent.futures as cf
from scipy import ndimage as ndi
from PIL import Image
B = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
d, zrel, out = sys.argv[1], sys.argv[2], sys.argv[3]
n_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 21
col0, col1 = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (None, None)
row0, row1 = (int(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) > 8 else (None, None)
cache = sys.argv[9] if len(sys.argv) > 9 else 'C:/Users/mdevi/prize/vesuvius/data/chunk_cache/' + zrel.strip('/').replace('/', '_')
os.makedirs(cache, exist_ok=True)
meta = json.load(open(os.path.join(d, 'meta.json'))); scale = float(meta['scale'][0]); step = int(round(1 / scale))
X = tifffile.imread(os.path.join(d, 'x.tif')).astype(np.float64); Y = tifffile.imread(os.path.join(d, 'y.tif')).astype(np.float64); Z = tifffile.imread(os.path.join(d, 'z.tif')).astype(np.float64)
valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & ~((X == 0) & (Y == 0) & (Z == 0)) & (X > -1) & (Y > -1) & (Z > -1)
X = np.where(valid, X, np.nan); Y = np.where(valid, Y, np.nan); Z = np.where(valid, Z, np.nan)
H, W = X.shape; print('grid', X.shape, 'step', step, 'valid frac', round(float(valid.mean()), 3), flush=True)
r0, r1 = (0, H) if row0 is None else (max(0, row0), min(H, row1)); c0, c1 = (0, W) if col0 is None else (max(0, col0), min(W, col1))
X, Y, Z, valid = X[r0:r1, c0:c1], Y[r0:r1, c0:c1], Z[r0:r1, c0:c1], valid[r0:r1, c0:c1]
h, w = X.shape
# fill NaNs for interpolation, then upsample coordinates to native spacing
def fill(a):
    m = np.isnan(a)
    if m.all(): return a
    idx = ndi.distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)]
Xf, Yf, Zf = fill(X), fill(Y), fill(Z)
Hn, Wn = (h - 1) * step + 1, (w - 1) * step + 1
zoom = ((Hn) / h, (Wn) / w)
Xu = ndi.zoom(Xf, zoom, order=1)[:Hn, :Wn]; Yu = ndi.zoom(Yf, zoom, order=1)[:Hn, :Wn]; Zu = ndi.zoom(Zf, zoom, order=1)[:Hn, :Wn]
Vu = ndi.zoom(valid.astype(np.float32), zoom, order=1)[:Hn, :Wn] > 0.5
# normals from tangents (row and column directions)
dr = np.stack(np.gradient(np.stack([Zu, Yu, Xu]), axis=1), 0)[:, 1] if False else None
tz_r, ty_r, tx_r = np.gradient(Zu, axis=0), np.gradient(Yu, axis=0), np.gradient(Xu, axis=0)
tz_c, ty_c, tx_c = np.gradient(Zu, axis=1), np.gradient(Yu, axis=1), np.gradient(Xu, axis=1)
nz = ty_r * tx_c - tx_r * ty_c; ny = tx_r * tz_c - tz_r * tx_c; nx = tz_r * ty_c - ty_r * tz_c
nn = np.sqrt(nz ** 2 + ny ** 2 + nx ** 2) + 1e-9; nz, ny, nx = nz / nn, ny / nn, nx / nn
print('native grid', (Hn, Wn), 'layers', n_layers, flush=True)
# gather required chunks
half = n_layers // 2
zs = Zu[None] + np.arange(-half, n_layers - half)[:, None, None] * nz[None]
ys = Yu[None] + np.arange(-half, n_layers - half)[:, None, None] * ny[None]
xs = Xu[None] + np.arange(-half, n_layers - half)[:, None, None] * nx[None]
c = 128
ci = np.stack([np.floor(zs / c), np.floor(ys / c), np.floor(xs / c)], -1).astype(np.int64)
mask3 = np.broadcast_to(Vu[None], zs.shape)
keys = np.unique(ci[mask3].reshape(-1, 3), axis=0)
print('chunks needed', len(keys), f'(~{len(keys)*2/1024:.1f} GB)', flush=True)
def fetch(k):
    kz, ky, kx = map(int, k); p = os.path.join(cache, f'{kz}_{ky}_{kx}')
    if os.path.exists(p) and os.path.getsize(p) == c ** 3: return 1
    if min(kz, ky, kx) < 0: return 0
    url = B + urllib.parse.quote(f'{zrel}0/{kz}/{ky}/{kx}')
    for a in range(5):
        try:
            buf = urllib.request.urlopen(url, timeout=120).read()
            if len(buf) == c ** 3: open(p, 'wb').write(buf); return 1
            return 0
        except urllib.error.HTTPError as e:
            if e.code == 404: open(p, 'wb').close(); return 0
            time.sleep(1 + a)
        except Exception: time.sleep(1 + a)
    return 0
t0 = time.time()
with cf.ThreadPoolExecutor(32) as ex: got = sum(ex.map(fetch, keys))
print('fetched', got, 'chunks in', round(time.time() - t0), 's', flush=True)
# sample: process chunk by chunk (nearest-voxel sampling for speed)
layers = np.zeros(zs.shape, np.uint8)
zi = np.clip(np.round(zs), 0, None).astype(np.int64); yi = np.clip(np.round(ys), 0, None).astype(np.int64); xi = np.clip(np.round(xs), 0, None).astype(np.int64)
flat_key = (zi // c) * (10 ** 8) + (yi // c) * (10 ** 4) + (xi // c)
uniq = np.unique(flat_key[mask3])
for fk in uniq:
    kz, ky, kx = int(fk // 10 ** 8), int((fk // 10 ** 4) % 10 ** 4), int(fk % 10 ** 4)
    p = os.path.join(cache, f'{kz}_{ky}_{kx}')
    if not (os.path.exists(p) and os.path.getsize(p) == c ** 3): continue
    blk = np.frombuffer(open(p, 'rb').read(), np.uint8).reshape(c, c, c)
    sel = (flat_key == fk) & mask3
    layers[sel] = blk[zi[sel] % c, yi[sel] % c, xi[sel] % c]
os.makedirs(out, exist_ok=True)
json.dump({"zarr_format": 2}, open(os.path.join(out, '.zgroup'), 'w'))
json.dump({"multiscales": [{"axes": [{"name": n, "type": "space"} for n in "zyx"], "datasets": [{"path": "0"}], "version": "0.4"}], "tifxyz": d, "rows": [r0, r1], "cols": [c0, c1]}, open(os.path.join(out, '.zattrs'), 'w'))
a_ = zarr.open(os.path.join(out, '0'), mode='w', shape=layers.shape, chunks=(n_layers, 512, 512), dtype='u1', zarr_format=2); a_[:] = layers
Image.fromarray(layers[half][::2, ::2]).save(out.rstrip('/').rstrip('\\') + '_mid_ds2.png')
print('wrote', out, layers.shape, 'nonzero frac', round(float((layers[half] > 0).mean()), 3), flush=True)
