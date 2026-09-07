"""Download an axis-aligned box (nz,ny,nx) of a public uncompressed level-0 zarr (128^3 chunks) into a local zarr-v2 store.
Usage: fetch_block3.py <zarr_rel_path/> z0 y0 x0 nz ny nx <out_dir> [level]"""
import sys, os, json, urllib.request, urllib.parse, concurrent.futures as cf, time
B = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
zpath = sys.argv[1]; z0, y0, x0, nz, ny, nx = map(int, sys.argv[2:8]); out = sys.argv[8]
level = sys.argv[9] if len(sys.argv) > 9 else '0'
c = 128; assert all(v % c == 0 for v in (z0, y0, x0, nz, ny, nx))
arr = os.path.join(out, '0'); os.makedirs(arr, exist_ok=True)
json.dump({"zarr_format": 2, "shape": [nz, ny, nx], "chunks": [c, c, c], "dtype": "|u1", "compressor": None, "fill_value": 0,
           "order": "C", "filters": None, "dimension_separator": "/"}, open(os.path.join(arr, '.zarray'), 'w'))
json.dump({"zarr_format": 2}, open(os.path.join(out, '.zgroup'), 'w'))
json.dump({"multiscales": [{"axes": [{"name": a, "type": "space"} for a in "zyx"], "datasets": [{"path": "0"}], "version": "0.4"}],
           "source": zpath, "origin_zyx": [z0, y0, x0], "level": level}, open(os.path.join(out, '.zattrs'), 'w'))
jobs = [(iz, iy, ix) for iz in range(nz // c) for iy in range(ny // c) for ix in range(nx // c)]
def get(j):
    iz, iy, ix = j; dst = os.path.join(arr, str(iz), str(iy), str(ix))
    if os.path.exists(dst) and os.path.getsize(dst) == c ** 3: return 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = B + urllib.parse.quote(f"{zpath}{level}/{z0//c+iz}/{y0//c+iy}/{x0//c+ix}")
    for k in range(5):
        try:
            buf = urllib.request.urlopen(url, timeout=60).read()
            if len(buf) != c ** 3: return 0
            open(dst, 'wb').write(buf); return 1
        except urllib.error.HTTPError as e:
            if e.code == 404: return 0
            time.sleep(1 + k)
        except Exception:
            time.sleep(1 + k)
    return 0
t0 = time.time(); done = 0
with cf.ThreadPoolExecutor(32) as ex:
    for i, r in enumerate(ex.map(get, jobs)):
        done += r
        if i % 500 == 499: print(f'{i+1}/{len(jobs)} chunks, {time.time()-t0:.0f}s', flush=True)
print(f"{done}/{len(jobs)} chunks present, {time.time()-t0:.0f}s ->", out, flush=True)
