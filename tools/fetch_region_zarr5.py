"""Box fetch from a compressed public zarr-v2 array (blosc, '/' separator) by direct chunk HTTP downloads with retries.
Usage: fetch_region_zarr5.py <array_url> z0 y0 x0 nz ny nx <out_dir>"""
import sys, os, json, time, urllib.request, numpy as np, zarr, numcodecs, concurrent.futures as cf
url = sys.argv[1].rstrip('/'); z0, y0, x0, nz, ny, nx = map(int, sys.argv[2:8]); out = sys.argv[8]
meta = json.loads(urllib.request.urlopen(url + '/.zarray', timeout=60).read().decode())
shape, chunks, dtype = meta['shape'], meta['chunks'], np.dtype(meta['dtype'])
codec = numcodecs.get_codec(meta['compressor']) if meta['compressor'] else None
sep = meta.get('dimension_separator', '.')
print('remote', shape, chunks, dtype, meta['compressor'] and meta['compressor']['id'], flush=True)
cz, cy, cx = chunks
vol = np.zeros((nz, ny, nx), dtype)
jobs = []
for iz in range(z0 // cz, (z0 + nz - 1) // cz + 1):
    for iy in range(y0 // cy, (y0 + ny - 1) // cy + 1):
        for ix in range(x0 // cx, (x0 + nx - 1) // cx + 1):
            jobs.append((iz, iy, ix))
def get(j):
    iz, iy, ix = j
    key = sep.join(map(str, (iz, iy, ix)))
    for k in range(6):
        try:
            buf = urllib.request.urlopen(f'{url}/{key}', timeout=120).read()
            arr = np.frombuffer(codec.decode(buf) if codec else buf, dtype).reshape(chunks)
            return j, arr
        except urllib.error.HTTPError as e:
            if e.code == 404: return j, None
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return j, 'FAIL'
t0 = time.time(); fails = 0
with cf.ThreadPoolExecutor(16) as ex:
    for (iz, iy, ix), arr in ex.map(get, jobs):
        if isinstance(arr, str): fails += 1; continue
        if arr is None: continue
        gz0, gy0, gx0 = iz * cz, iy * cy, ix * cx
        # overlap between chunk box and requested box
        z_lo, z_hi = max(gz0, z0), min(gz0 + cz, z0 + nz); y_lo, y_hi = max(gy0, y0), min(gy0 + cy, y0 + ny); x_lo, x_hi = max(gx0, x0), min(gx0 + cx, x0 + nx)
        if z_lo >= z_hi or y_lo >= y_hi or x_lo >= x_hi: continue
        vol[z_lo - z0:z_hi - z0, y_lo - y0:y_hi - y0, x_lo - x0:x_hi - x0] = arr[z_lo - gz0:z_hi - gz0, y_lo - gy0:y_hi - gy0, x_lo - gx0:x_hi - gx0]
print(f'{len(jobs)} chunks, {fails} failed, {time.time()-t0:.0f}s', flush=True)
if fails: raise SystemExit('FAILED chunks')
os.makedirs(out, exist_ok=True)
json.dump({"zarr_format": 2}, open(os.path.join(out, '.zgroup'), 'w'))
json.dump({"origin_zyx": [z0, y0, x0], "source": url, "complete": True}, open(os.path.join(out, '.zattrs'), 'w'))
dst = zarr.open(os.path.join(out, '0'), mode='w', shape=vol.shape, chunks=(128, 128, 128), dtype=vol.dtype, zarr_format=2); dst[:] = vol
print('wrote', out, vol.shape, 'mean', float(vol.mean()), 'frac>0', float((vol > 0).mean()), flush=True)
