"""Model-free views of a rendered sheet stack (C,H,W): per-layer mean profile, surface-vs-interior difference maps, and a local
contrast-normalised 'crackle' map. Usage: layer_stats.py <sheet.zarr> <out_prefix> [ds=2]"""
import sys, os, numpy as np, zarr
from scipy import ndimage as ndi
from PIL import Image
src, out = sys.argv[1], sys.argv[2]; ds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
g = zarr.open(src, mode='r'); a = np.asarray(g['0'] if hasattr(g, 'keys') and '0' in g else g)
if a.shape[0] != min(a.shape): a = np.transpose(a, (2, 0, 1))
C, H, W = a.shape
mask = a[C // 2] > 0
prof = [float(a[c][mask].mean()) if mask.any() else 0 for c in range(C)]
print('layer mean profile:', ' '.join('%.0f' % p for p in prof))
core = int(np.argmax(prof))   # densest layer ~ sheet centre
print('densest layer', core, 'of', C)
def save(im, name):
    im = im[::ds, ::ds]; lo, hi = np.percentile(im[im != 0], [1, 99]) if (im != 0).any() else (0, 1)
    Image.fromarray(np.clip((im - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)).save(out + '_' + name + '.png')
f = a.astype(np.float32)
for name, sl in [('surfA', slice(max(0, core - 12), max(1, core - 6))), ('surfB', slice(min(C - 1, core + 6), min(C, core + 12))), ('core', slice(max(0, core - 2), min(C, core + 3)))]:
    m = f[sl].mean(0)
    save(m, 'mean_' + name)
    # local contrast normalisation at ~0.4 mm scale
    bg = ndi.gaussian_filter(m, 80); sd = np.sqrt(ndi.gaussian_filter((m - bg) ** 2, 80)) + 1e-3
    save(np.where(mask, (m - bg) / sd, 0), 'lcn_' + name)
d = f[max(0, core - 10):max(1, core - 4)].mean(0) - f[max(0, core - 2):min(C, core + 3)].mean(0)
save(np.where(mask, ndi.gaussian_filter(d, 6), 0), 'diff_surfA_minus_core')
print('wrote', out)
