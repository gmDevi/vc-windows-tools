"""Robust box fetch from a compressed public zarr over HTTP: slab-wise with retries and long timeouts.
Usage: fetch_region_zarr4.py <url> z0 y0 x0 nz ny nx <out_dir>"""
import sys, os, json, time, numpy as np, zarr, fsspec, aiohttp
url = sys.argv[1]; z0, y0, x0, nz, ny, nx = map(int, sys.argv[2:8]); out = sys.argv[8]
def opener():
    m = fsspec.get_mapper(url, client_kwargs={'timeout': aiohttp.ClientTimeout(total=900, sock_read=300)})
    return zarr.open(m, mode='r')
a = opener(); print('remote', a.shape, a.chunks, a.dtype, flush=True)
os.makedirs(out, exist_ok=True)
json.dump({"zarr_format": 2}, open(os.path.join(out, '.zgroup'), 'w'))
json.dump({"origin_zyx": [z0, y0, x0], "source": url}, open(os.path.join(out, '.zattrs'), 'w'))
dst = zarr.open(os.path.join(out, '0'), mode='w', shape=(nz, ny, nx), chunks=(128, 128, 128), dtype=a.dtype, zarr_format=2)
step = 64
for z in range(0, nz, step):
    zz = min(nz, z + step)
    for attempt in range(8):
        try:
            dst[z:zz] = np.asarray(a[z0 + z:z0 + zz, y0:y0 + ny, x0:x0 + nx]); break
        except Exception as e:
            print('retry', z, attempt, type(e).__name__, flush=True); time.sleep(5 * (attempt + 1)); a = opener()
    else:
        raise SystemExit('FAILED at z=%d' % z)
    print('slab', z, zz, flush=True)
json.dump({"origin_zyx": [z0, y0, x0], "source": url, "complete": True}, open(os.path.join(out, '.zattrs'), 'w'))
print('wrote', out, (nz, ny, nx), flush=True)
