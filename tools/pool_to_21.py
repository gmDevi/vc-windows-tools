"""Take a 109-layer ~9.6um surface-volume crop, keep the centered 84 planes, mean-pool every 4 -> 21 slices, write uncompressed zarr-v2 group for flat inference."""
import sys, os, json, numpy as np, zarr
src, dst = sys.argv[1], sys.argv[2]
a = zarr.open(src, mode='r'); a = a['0'] if hasattr(a, 'keys') and '0' in a else a
L = a.shape[0]; start = (L - 84) // 2
v = np.asarray(a[start:start+84]).astype(np.float32).reshape(21, 4, a.shape[1], a.shape[2]).mean(axis=1).astype(np.uint8)
os.makedirs(dst, exist_ok=True)
json.dump({"zarr_format": 2}, open(os.path.join(dst, '.zgroup'), 'w'))
json.dump({"multiscales": [{"axes": [{"name": n, "type": "space"} for n in "zyx"], "datasets": [{"path": "0"}], "version": "0.4"}]}, open(os.path.join(dst, '.zattrs'), 'w'))
z = zarr.open(os.path.join(dst, '0'), mode='w', shape=v.shape, chunks=(21, 512, 512), dtype='u1', zarr_format=2)
z[:] = v; print('wrote', dst, v.shape, 'mean', v.mean())
