"""Estimate the spiral outward sense (CW/ACW, visual, y-down image coords) from one cross-section of the surface prediction.
Polar-transform around the umbilicus; sheets become lines r(theta); the sign of the dominant slope dr/dtheta gives the sense."""
import sys, json, numpy as np, zarr, fsspec
from scipy import ndimage as ndi
from PIL import Image
scroll, vol, surf_rel, umb_url, z0 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
B = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
umb = json.loads(fsspec.open(B + umb_url).open().read())['control_points']
zs = np.array([p['z'] for p in umb]); xs = np.array([p['x'] for p in umb]); ys = np.array([p['y'] for p in umb])
o = np.argsort(zs); cx = np.interp(z0, zs[o], xs[o]); cy = np.interp(z0, zs[o], ys[o])
lvl = 2; f = 4
a = zarr.open(fsspec.get_mapper(B + surf_rel + str(lvl)), mode='r')
sl = np.asarray(a[z0 // f]).astype(np.float32); print('slice', sl.shape, 'umbilicus (x,y) level0', round(cx), round(cy), flush=True)
cxl, cyl = cx / f, cy / f
Image.fromarray(np.clip(sl, 0, 255).astype(np.uint8)).save(f'C:/Users/mdevi/prize/vesuvius/preds/{scroll}_surf_z{z0}.png')
# polar transform: rows = theta (visual clockwise, y-down), cols = r
nr, nt = 1200, 1440
rmax = min(cxl, cyl, sl.shape[1] - cxl, sl.shape[0] - cyl) * 0.98
rr = np.linspace(2, rmax, nr); tt = np.linspace(-np.pi, np.pi, nt, endpoint=False)
R, T = np.meshgrid(rr, tt, indexing='ij')
X = cxl + R * np.cos(T); Y = cyl + R * np.sin(T)
pol = ndi.map_coordinates(sl, [Y.ravel(), X.ravel()], order=1, mode='constant').reshape(nr, nt)  # [r, theta]
Image.fromarray(np.clip(pol, 0, 255).astype(np.uint8)).save(f'C:/Users/mdevi/prize/vesuvius/preds/{scroll}_polar_z{z0}.png')
# structure tensor orientation of sheet lines in (r, theta) space
g = ndi.gaussian_filter(pol, 1.5)
gr, gt = np.gradient(g)                     # d/dr, d/dtheta(index)
w = (pol > 60).astype(np.float32)
# line direction is perpendicular to gradient; slope dr/dtheta of the line = -gt/gr
Jrr = ndi.gaussian_filter(gr * gr * w, 6); Jtt = ndi.gaussian_filter(gt * gt * w, 6); Jrt = ndi.gaussian_filter(gr * gt * w, 6)
# orientation angle of dominant gradient: phi = 0.5*atan2(2Jrt, Jrr-Jtt); line slope sign = -sign(Jrt) when |Jrr|>|Jtt| typical
slope = -Jrt / (Jrr + 1e-6)
m = (w > 0) & (Jrr > np.percentile(Jrr[w > 0], 50))
pos = float((slope[m] > 0).mean()); med = float(np.median(slope[m]))
print(f'fraction of sheet pixels with dr/dtheta>0: {pos:.3f}; median slope {med:.4f} (r-px per theta-index)')
print('SENSE:', 'CW' if med > 0 else 'ACW', '(visual clockwise = atan2(y-cy, x-cx) increasing, y down)')
