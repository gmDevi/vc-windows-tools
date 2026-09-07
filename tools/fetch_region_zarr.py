"""Read a region of a (possibly compressed) public zarr array over HTTP and save it as a local uncompressed zarr-v2 group ('0')."""
import sys, os, json, numpy as np, zarr, fsspec
url, z0, y0, x0, n, out = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6]
store = fsspec.get_mapper(url)
a = zarr.open(store, mode='r')
print('remote', a.shape, a.chunks, a.dtype, flush=True)
v = np.asarray(a[z0:z0+n, y0:y0+n, x0:x0+n])
os.makedirs(out, exist_ok=True)
json.dump({"zarr_format": 2}, open(os.path.join(out, '.zgroup'), 'w'))
json.dump({"origin_zyx": [z0, y0, x0], "source": url}, open(os.path.join(out, '.zattrs'), 'w'))
za = zarr.open(os.path.join(out, '0'), mode='w', shape=v.shape, chunks=(128, 128, 128), dtype=v.dtype, zarr_format=2)
za[:] = v
print('wrote', out, v.shape, 'mean', float(v.mean()), 'frac>0', float((v > 0).mean()), flush=True)
