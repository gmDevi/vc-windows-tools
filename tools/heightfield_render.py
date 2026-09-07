"""Poor-man's segmentation + rendering for the flat ink model.
Inputs: a local zarr block of the CT volume (uint8) and a matching block of the organizers' surface prediction (uint8 prob, same shape).
Steps: (1) rotate both about z so sheets are ~parallel to the yz plane (normal along x);
       (2) find sheet surfaces per (z,y) column as peaks of the surface prediction along x; region-grow one sheet from a seed into a
           height field x=f(z,y); smooth; (3) sample 21 layers at f(z,y)+k (k=-10..10) with linear interpolation; write zarr for infer.
Usage: heightfield_render.py <ct.zarr> <surf.zarr> <out_dir> [n_sheets]
"""
import sys, os, json, numpy as np, zarr
from scipy import ndimage as ndi
from PIL import Image
ct_path, surf_path, out = sys.argv[1:4]
n_sheets = int(sys.argv[4]) if len(sys.argv) > 4 else 6
os.makedirs(out, exist_ok=True)
def load(p):
    g = zarr.open(p, mode='r'); return np.asarray(g['0'] if hasattr(g, 'keys') and '0' in g else g)
ct = load(ct_path); surf = load(surf_path)
assert ct.shape == surf.shape, (ct.shape, surf.shape)
# (1) dominant normal in xy from CT gradients
cov = np.zeros((2, 2))
for z in range(0, ct.shape[0], max(1, ct.shape[0] // 8)):
    s = ndi.gaussian_filter(ct[z].astype(np.float32), 2); dy, dx = np.gradient(s)
    cov += np.array([[(dy*dy).sum(), (dy*dx).sum()], [(dy*dx).sum(), (dx*dx).sum()]])
w, v = np.linalg.eigh(cov); nrm = v[:, np.argmax(w)]; theta = np.degrees(np.arctan2(nrm[0], nrm[1]))
def aniso(img):
    dy, dx = np.gradient(ndi.gaussian_filter(img.astype(np.float32), 1.5)); return np.abs(dx).mean() / (np.abs(dy).mean() + 1e-6)
mid = ct[ct.shape[0] // 2]
ang = max((theta, -theta, theta + 90, -theta + 90, theta - 90, -theta - 90), key=lambda a: aniso(ndi.rotate(mid, a, reshape=False, order=1)))
print('rotation', round(ang, 1), flush=True)
rct = ndi.rotate(ct, ang, axes=(1, 2), reshape=False, order=1).astype(np.float32)
rsf = ndi.rotate(surf, ang, axes=(1, 2), reshape=False, order=1).astype(np.float32)
Z, Y, X = rct.shape
Image.fromarray(np.clip(rct[Z//2], 0, 255).astype(np.uint8)).save(os.path.join(out, 'rot_midz_ct.png'))
Image.fromarray(np.clip(rsf[Z//2], 0, 255).astype(np.uint8)).save(os.path.join(out, 'rot_midz_surf.png'))
# (2) per-column peaks of the surface prediction along x (smoothed)
sm = ndi.gaussian_filter(rsf, (2, 2, 1))
peak = (sm > 60) & (sm >= np.roll(sm, 1, axis=2)) & (sm >= np.roll(sm, -1, axis=2))
# region-grow height fields: seeds = strongest peaks on the mid-column line, grown by nearest-peak continuity
occupied = np.zeros_like(peak)
sheets = []
zc, yc = Z // 2, Y // 2
cand_x = np.where(peak[zc, yc])[0]
cand_x = sorted(cand_x, key=lambda x: -sm[zc, yc, x])
for x0 in cand_x:
    if len(sheets) >= n_sheets: break
    if occupied[zc, yc, max(0, x0-3):x0+4].any(): continue
    f = np.full((Z, Y), np.nan, np.float32); f[zc, yc] = x0
    # BFS over (z,y) grid with step 4 for speed, then upsample
    step = 4
    fz, fy = np.arange(0, Z, step), np.arange(0, Y, step)
    F = np.full((len(fz), len(fy)), np.nan, np.float32)
    iz0, iy0 = zc // step, yc // step; F[iz0, iy0] = x0
    from collections import deque
    q = deque([(iz0, iy0)]); visited = np.zeros_like(F, bool); visited[iz0, iy0] = True
    while q:
        iz, iy = q.popleft(); xprev = F[iz, iy]
        for dz, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            jz, jy = iz+dz, iy+dy
            if jz < 0 or jy < 0 or jz >= len(fz) or jy >= len(fy) or visited[jz, jy]: continue
            visited[jz, jy] = True
            zz, yy = fz[jz], fy[jy]; lo, hi = int(max(0, xprev-6)), int(min(X, xprev+7))
            xs = np.where(peak[zz, yy, lo:hi])[0]
            if xs.size == 0: continue
            xn = lo + xs[np.argmin(np.abs(lo + xs - xprev))]
            F[jz, jy] = xn; q.append((jz, jy))
    valid = ~np.isnan(F)
    if valid.mean() < 0.25: continue
    # fill holes + smooth
    Ff = F.copy(); Ff[~valid] = 0
    num = ndi.gaussian_filter(Ff, 3); den = ndi.gaussian_filter(valid.astype(np.float32), 3)
    Fs = np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan)
    full = ndi.zoom(np.nan_to_num(Fs, nan=np.nanmean(Fs)), (Z / len(fz), Y / len(fy)), order=1)
    vmask = ndi.zoom(valid.astype(np.float32), (Z / len(fz), Y / len(fy)), order=1) > 0.5
    for (jz, jy) in zip(*np.where(valid)):
        xx = int(round(F[jz, jy])); occupied[fz[jz], fy[jy], max(0, xx-3):xx+4] = True
    sheets.append((x0, full, vmask, valid.mean()))
    print(f'sheet seed x={x0}: coverage {valid.mean():.2f}', flush=True)
# (3) sample 21 layers along x at f + k
zz, yy = np.meshgrid(np.arange(Z), np.arange(Y), indexing='ij')
for i, (x0, full, vmask, cov_) in enumerate(sheets):
    layers = np.zeros((21, Z, Y), np.uint8)
    for k in range(-10, 11):
        coords = np.stack([zz.ravel(), yy.ravel(), (full + k).ravel()])
        vals = ndi.map_coordinates(rct, coords, order=1, mode='constant', cval=0).reshape(Z, Y)
        layers[k + 10] = np.clip(vals, 0, 255).astype(np.uint8) * vmask
    name = f'sheet{i}_x{x0}'
    d = os.path.join(out, name + '.zarr'); os.makedirs(d, exist_ok=True)
    json.dump({"zarr_format": 2}, open(os.path.join(d, '.zgroup'), 'w'))
    json.dump({"multiscales": [{"axes": [{"name": n, "type": "space"} for n in "zyx"], "datasets": [{"path": "0"}], "version": "0.4"}]}, open(os.path.join(d, '.zattrs'), 'w'))
    a = zarr.open(os.path.join(d, '0'), mode='w', shape=layers.shape, chunks=(21, 512, 512), dtype='u1', zarr_format=2); a[:] = layers
    Image.fromarray(layers[10]).save(os.path.join(out, name + '_mid.png'))
    print('wrote', d, 'coverage', round(cov_, 2), flush=True)
