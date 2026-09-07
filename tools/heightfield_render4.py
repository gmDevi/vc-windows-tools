"""Tilted-sheet renderer for big blocks: rotate only the small L2 surface prediction to make sheets vertical, grow height fields
there, then sample the ORIGINAL (unrotated) CT by inverse-rotating sample coordinates. No big-volume rotation.
Usage: heightfield_render4.py <ct.zarr (Z,Y,X uint8 L0)> <surfL2.zarr (same box at 1/4)> <out_dir> [n_sheets=4] [n_layers=62] [S=4]
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
ct = load(ct_path); surf = load(surf_path).astype(np.float32)
Z, Y, X = ct.shape; Zs, Ys, Xs = surf.shape; print('ct', ct.shape, 'surfL2', surf.shape, flush=True)

# --- dominant sheet normal angle in the (y,x) plane from the L2 surface prediction (structure tensor)
cov = np.zeros((2, 2))
for z in range(0, Zs, max(1, Zs // 8)):
    s = ndi.gaussian_filter(surf[z], 1.5); dy, dx = np.gradient(s)
    cov += np.array([[(dy * dy).sum(), (dy * dx).sum()], [(dy * dx).sum(), (dx * dx).sum()]])
w, v = np.linalg.eigh(cov); nrm = v[:, np.argmax(w)]
theta = np.degrees(np.arctan2(nrm[0], nrm[1]))
def aniso(img):
    dy, dx = np.gradient(ndi.gaussian_filter(img, 1.5)); return np.abs(dx).mean() / (np.abs(dy).mean() + 1e-6)
mid = surf[Zs // 2]
ang = max((theta, -theta, theta + 90, -theta + 90, theta - 90, -theta - 90), key=lambda a: aniso(ndi.rotate(mid, a, reshape=False, order=1)))
print('rotation %.1f deg (normal -> +x in rotated frame)' % ang, flush=True)
rs = ndi.rotate(surf, ang, axes=(1, 2), reshape=True, order=1)          # rotated L2 surface, sheets ~vertical
_, Yr, Xr = rs.shape
Image.fromarray(np.clip(rs[Zs // 2], 0, 255).astype(np.uint8)).save(os.path.join(out, 'rotL2_midz_surf.png'))
# rotation mapping: scipy rotates about the image center; build inverse map (rotated (yr,xr) -> original (y,x)) for L2 coords
a = np.deg2rad(ang); c, s_ = np.cos(a), np.sin(a)
cy_o, cx_o = (Ys - 1) / 2.0, (Xs - 1) / 2.0; cy_r, cx_r = (Yr - 1) / 2.0, (Xr - 1) / 2.0
def rot_to_orig(yr, xr):
    # scipy.ndimage.rotate(angle) maps output coords via inverse rotation; empirically calibrate direction below
    dy, dx = yr - cy_r, xr - cx_r
    y = cy_o + c * dy - s_ * dx; x = cx_o + s_ * dy + c * dx
    return y, x
# calibrate the sign convention by checking which mapping reproduces rs from surf at the mid slice
yy, xx = np.meshgrid(np.arange(Yr), np.arange(Xr), indexing='ij')
best = None
for sign in (1, -1):
    dy, dx = yy - cy_r, xx - cx_r
    y = cy_o + c * dy - sign * s_ * dx; x = cx_o + sign * s_ * dy + c * dx
    rec = ndi.map_coordinates(surf[Zs // 2], [y.ravel(), x.ravel()], order=1, mode='constant').reshape(Yr, Xr)
    err = np.abs(rec - rs[Zs // 2]).mean()
    if best is None or err < best[0]: best = (err, sign)
sign = best[1]; print('rotation sign', sign, 'recon err %.2f' % best[0], flush=True)
def rot_to_orig(yr, xr):
    dy, dx = yr - cy_r, xr - cx_r
    return cy_o + c * dy - sign * s_ * dx, cx_o + sign * s_ * dy + c * dx

# --- height fields in the rotated L2 frame
sm = ndi.gaussian_filter(rs, (1.5, 1.5, 0.7))
peak = (sm > 60) & (sm >= np.roll(sm, 1, axis=2)) & (sm >= np.roll(sm, -1, axis=2))
occupied = np.zeros_like(peak); zc, yc = Zs // 2, Yr // 2
cands = sorted(np.where(peak[zc, yc])[0], key=lambda x: -sm[zc, yc, x])
sheets = []
for x0 in cands:
    if len(sheets) >= n_sheets: break
    if occupied[zc, yc, max(0, x0 - 2):x0 + 3].any(): continue
    F = np.full((Zs, Yr), np.nan, np.float32); F[zc, yc] = x0
    q = deque([(zc, yc)]); visited = np.zeros((Zs, Yr), bool); visited[zc, yc] = True
    while q:
        iz, iy = q.popleft(); xprev = F[iz, iy]
        for dz, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            jz, jy = iz + dz, iy + dy
            if jz < 0 or jy < 0 or jz >= Zs or jy >= Yr or visited[jz, jy]: continue
            visited[jz, jy] = True
            lo, hi = int(max(0, xprev - 3)), int(min(Xr, xprev + 4))
            xs = np.where(peak[jz, jy, lo:hi])[0]
            if xs.size == 0: continue
            F[jz, jy] = lo + xs[np.argmin(np.abs(lo + xs - xprev))]; q.append((jz, jy))
    valid = ~np.isnan(F); covr = valid.mean()
    if covr < 0.15: continue
    Ff = np.where(valid, F, 0).astype(np.float32)
    num = ndi.gaussian_filter(Ff, 2); den = ndi.gaussian_filter(valid.astype(np.float32), 2)
    Fs = np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan); Fs = np.where(np.isnan(Fs), np.nanmean(Fs), Fs)
    for (jz, jy) in zip(*np.where(valid)):
        xx0 = int(round(F[jz, jy])); occupied[jz, jy, max(0, xx0 - 2):xx0 + 3] = True
    sheets.append((int(x0), Fs, valid, float(covr))); print(f'sheet seed xr={x0}: coverage {covr:.2f}', flush=True)

# --- sample layers from the ORIGINAL CT at level 0: rotated-frame sample (z, yr, xr=f+k) -> original (y,x) -> *S
half = n_layers // 2
Zf, Yf = Z, Yr * S
zz, yy = np.meshgrid(np.arange(Zf), np.arange(Yf), indexing='ij')
for i, (x0, Fs, valid, covr) in enumerate(sheets):
    full = ndi.zoom(Fs, (Zf / Zs, Yf / Yr), order=1)[:Zf, :Yf]                     # xr at L2 units
    vmask = ndi.zoom(valid.astype(np.float32), (Zf / Zs, Yf / Yr), order=1)[:Zf, :Yf] > 0.5
    layers = np.zeros((n_layers, Zf, Yf), np.uint8)
    for k in range(-half, n_layers - half):
        yr = yy / S; xr = full + k / S                                                  # rotated L2 coords
        yo, xo = rot_to_orig(yr, xr)                                                    # original L2 coords
        coords = np.stack([zz.ravel(), (yo * S + (S - 1) / 2).ravel(), (xo * S + (S - 1) / 2).ravel()])
        vals = ndi.map_coordinates(ct, coords, order=1, mode='constant', cval=0).reshape(Zf, Yf)
        layers[k + half] = np.clip(vals, 0, 255).astype(np.uint8) * vmask
    inside = layers[half] > 0
    name = f'sheet{i}_xr{x0}'
    d = os.path.join(out, name + '.zarr'); os.makedirs(d, exist_ok=True)
    json.dump({"zarr_format": 2}, open(os.path.join(d, '.zgroup'), 'w'))
    json.dump({"multiscales": [{"axes": [{"name": n, "type": "space"} for n in "zyx"], "datasets": [{"path": "0"}], "version": "0.4"}],
               "rotation_deg": float(ang), "coverage": covr, "inside_frac": float(inside.mean())}, open(os.path.join(d, '.zattrs'), 'w'))
    a_ = zarr.open(os.path.join(d, '0'), mode='w', shape=layers.shape, chunks=(n_layers, 512, 512), dtype='u1', zarr_format=2); a_[:] = layers
    Image.fromarray(layers[half][::2, ::2]).save(os.path.join(out, name + '_mid_ds2.png'))
    print('wrote', d, 'coverage', round(covr, 2), 'inside', round(float(inside.mean()), 2), flush=True)
