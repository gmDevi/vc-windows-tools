"""Robust re-scoring of sweep outputs from their preview PNGs (ink map + mid-layer render, both at ds2).
A bright blob only counts if it lies on textured papyrus (local std of the render above a threshold), does not touch the
render mask boundary, and has a letter-like size (at ds2: 100-7500 px, i.e. 400-30000 px at full res).
Usage: rescore.py <results_dir> [...]   -> prints a table and writes rescored.json in each dir."""
import sys, os, glob, json, numpy as np
from PIL import Image
from scipy import ndimage as ndi
def score_pair(ink_png, render_png):
    a = np.array(Image.open(ink_png).convert('L')).astype(np.float32)
    r = np.array(Image.open(render_png).convert('L')).astype(np.float32)
    h, w = min(a.shape[0], r.shape[0]), min(a.shape[1], r.shape[1]); a, r = a[:h, :w], r[:h, :w]
    inside = r > 0
    inside_er = ndi.binary_erosion(inside, iterations=12)                     # away from mask boundary (24 px at full res)
    m = ndi.uniform_filter(r, 15); m2 = ndi.uniform_filter(r * r, 15); lstd = np.sqrt(np.maximum(m2 - m * m, 0))
    textured = lstd > 6                                                        # papyrus fibre texture, not flat fill
    hi = (a > 200) & inside_er & textured
    lab, n = ndi.label(hi)
    sizes = ndi.sum(hi, lab, range(1, n + 1)) if n else np.array([])
    letter = int(((sizes > 100) & (sizes < 7500)).sum())
    return {'frac200_inside': float(((a > 200) & inside_er).sum() / max(1, inside_er.sum())), 'blobs_letterlike_robust': letter,
            'inside_frac': float(inside.mean()), 'textured_frac': float((textured & inside).sum() / max(1, inside.sum()))}
for d in sys.argv[1:]:
    out = {}
    for ink in sorted(glob.glob(os.path.join(d, '*_ink.png'))):
        base = os.path.basename(ink)[:-8]
        render = None
        for cand in (os.path.join(d, base + '.zarr_mid_ds2.png'), os.path.join(d, base + '_mid.png'), os.path.join(d, base + '_mid_ds2.png')):
            if os.path.exists(cand): render = cand; break
        if render is None: continue
        out[base] = score_pair(ink, render)
        rev = ink[:-4] + '_reverse.png'
        if os.path.exists(rev): out[base + '_reverse'] = score_pair(rev, render)
    json.dump(out, open(os.path.join(d, 'rescored.json'), 'w'), indent=1)
    top = sorted(out.items(), key=lambda kv: -kv[1]['blobs_letterlike_robust'])[:6]
    print(os.path.basename(d.rstrip('/\\')), '| n=%d | top robust blobs:' % len(out), ', '.join(f"{k}:{v['blobs_letterlike_robust']} (frac {v['frac200_inside']:.4f})" for k, v in top))
