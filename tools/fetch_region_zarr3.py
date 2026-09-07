"""Read a box (nz,ny,nx) of a compressed public zarr array over HTTP and save as local uncompressed zarr-v2 group ('0')."""
import sys, os, json, numpy as np, zarr, fsspec
url = sys.argv[1]; z0, y0, x0, nz, ny, nx = map(int, sys.argv[2:8]); out = sys.argv[8]
a = zarr.open(fsspec.get_mapper(url), mode='r'); print('remote', a.shape, a.chunks, a.dtype, flush=True)
v = np.asarray(a[z0:z0+nz, y0:y0+ny, x0:x0+nx])
os.makedirs(out, exist_ok=True)
json.dump({"zarr_format": 2}, open(os.path.join(out, '.zgroup'), 'w'))
json.dump({"origin_zyx": [z0, y0, x0], "source": url}, open(os.path.join(out, '.zattrs'), 'w'))
za = zarr.open(os.path.join(out, '0'), mode='w', shape=v.shape, chunks=(128, 128, 128), dtype=v.dtype, zarr_format=2); za[:] = v
print('wrote', out, v.shape, 'mean', float(v.mean()), 'frac>0', float((v > 0).mean()), flush=True)
