"""Big-block sheet renderer for the 2um canonical model.
CT block (uint8, level 0, shape (Z,Y,X)) + organizers' surface prediction box at level L2 (same box, 4x coarser).
Sheets are assumed roughly parallel to one axis plane; the normal axis (y or x) is chosen from gradient energy on a mid slice,
and the block is permuted so the normal is the last axis. Height fields are grown at L2, upsampled 4x, and n_layers layers
are sampled along the normal by linear interpolation.
Usage: heightfield_render3.py <ct.zarr> <surfL2.zarr> <out_dir> [n_sheets=4] [n_layers=62] [surf_scale=4]
"""
import sys, os, json, numpy as np, zarr
from collections import deque
from scipy import ndimage as ndi
from PIL import Image

ct_path, surf_path, out = sys.argv[1:4]
n_sheets = int(sys.argv[4]) if len(sys.argv) > 4 else 4
n_layers = int(sys.argv[5]) if len(sys.argv) > 5 else 62
S = int(sys.argv[6]) if len(sys.argv) > 6 else 4
os.makedirs(out, exist_ok=True)

def load(p):
    g = zarr.open(p, mode='r'); return np.asarray(g['0'] if hasattr(g, 'keys') and '0' in g else g)

ct = load(ct_path); surf = load(surf_path)
print('ct', ct.shape, 'surf', surf.shape, flush=True)
assert all(abs(c // S - s) <= 1 for c, s in zip(ct.shape, surf.shape)), (ct.shape, surf.shape)

# --- choose the normal axis from a mid-z slice of the CT: larger mean |gradient| along an axis => normal along it
mid = ndi.gaussian_filter(ct[ct.shape[0] // 2].astype(np.float32), 2)
gy, gx = np.gradient(mid)
normal_axis = 1 if np.abs(gy).mean() > np.abs(gx).mean() else 2     # 1 = y, 2 = x
print('normal axis:', 'y' if normal_axis == 1 else 'x', '(|gy|=%.2f |gx|=%.2f)' % (np.abs(gy).mean(), np.abs(gx).mean()), flush=True)
if normal_axis == 1:
    ct = np.ascontiguousarray(np.swapaxes(ct, 1, 2)); surf = np.ascontiguousarray(np.swapaxes(surf, 1, 2))
Z, Y, X = ct.shape; Zs, Ys, Xs = surf.shape
Image.fromarray(ct[Z // 2]).save(os.path.join(out, 'perm_midz_ct.png'))
Image.fromarray(surf[Zs // 2]).save(os.path.join(out, 'perm_midz_surf.png'))

# --- sheet height fields at L2: peaks of the smoothed surface prediction along x
sm = ndi.gaussian_filter(surf.astype(np.float32), (1.5, 1.5, 0.7))
peak = (sm > 60) & (sm >= np.roll(sm, 1, axis=2)) & (sm >= np.roll(sm, -1, axis=2))
occupied = np.zeros_like(peak)
zc, yc = Zs // 2, Ys // 2
cands = sorted(np.where(peak[zc, yc])[0], key=lambda x: -sm[zc, yc, x])
sheets = []
for x0 in cands:
    if len(sheets) >= n_sheets: break
    if occupied[zc, yc, max(0, x0 - 2):x0 + 3].any(): continue
    F = np.full((Zs, Ys), np.nan, np.float32); F[zc, yc] = x0
    q = deque([(zc, yc)]); visited = np.zeros((Zs, Ys), bool); visited[zc, yc] = True
    while q:
        iz, iy = q.popleft(); xprev = F[iz, iy]
        for dz, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            jz, jy = iz + dz, iy + dy
            if jz < 0 or jy < 0 or jz >= Zs or jy >= Ys or visited[jz, jy]: continue
            visited[jz, jy] = True
            lo, hi = int(max(0, xprev - 3)), int(min(Xs, xprev + 4))
            xs = np.where(peak[jz, jy, lo:hi])[0]
            if xs.size == 0: continue
            F[jz, jy] = lo + xs[np.argmin(np.abs(lo + xs - xprev))]; q.append((jz, jy))
    valid = ~np.isnan(F)
    cov = valid.mean()
    if cov < 0.2: continue
    Ff = np.where(valid, F, 0).astype(np.float32)
    num = ndi.gaussian_filter(Ff, 2); den = ndi.gaussian_filter(valid.astype(np.float32), 2)
    Fs = np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan)
    Fs = np.where(np.isnan(Fs), np.nanmean(Fs), Fs)
    for (jz, jy) in zip(*np.where(valid)):
        xx = int(round(F[jz, jy])); occupied[jz, jy, max(0, xx - 2):xx + 3] = True
    sheets.append((int(x0), Fs, valid, float(cov)))
    print(f'sheet seed xL2={x0}: coverage {cov:.2f}', flush=True)

# --- sample layers at level 0
half = n_layers // 2
zz, yy = np.meshgrid(np.arange(Z), np.arange(Y), indexing='ij')
for i, (x0, Fs, valid, cov) in enumerate(sheets):
    full = ndi.zoom(Fs, (Z / Zs, Y / Ys), order=1)[:Z, :Y] * S + (S - 1) / 2.0
    vmask = ndi.zoom(valid.astype(np.float32), (Z / Zs, Y / Ys), order=1)[:Z, :Y] > 0.5
    layers = np.zeros((n_layers, Z, Y), np.uint8)
    for k in range(-half, n_layers - half):
        coords = np.stack([zz.ravel(), yy.ravel(), (full + k).ravel()])
        vals = ndi.map_coordinates(ct, coords, order=1, mode='constant', cval=0).reshape(Z, Y)
        layers[k + half] = np.clip(vals, 0, 255).astype(np.uint8) * vmask
    name = f'sheet{i}_xL2_{x0}'
    d = os.path.join(out, name + '.zarr'); os.makedirs(d, exist_ok=True)
    json.dump({"zarr_format": 2}, open(os.path.join(d, '.zgroup'), 'w'))
    json.dump({"multiscales": [{"axes": [{"name": n, "type": "space"} for n in "zyx"], "datasets": [{"path": "0"}], "version": "0.4"}],
               "normal_axis": 'y' if normal_axis == 1 else 'x', "coverage": cov}, open(os.path.join(d, '.zattrs'), 'w'))
    a = zarr.open(os.path.join(d, '0'), mode='w', shape=layers.shape, chunks=(n_layers, 512, 512), dtype='u1', zarr_format=2); a[:] = layers
    Image.fromarray(layers[half][::2, ::2]).save(os.path.join(out, name + '_mid_ds2.png'))
    print('wrote', d, 'coverage', round(cov, 2), flush=True)
