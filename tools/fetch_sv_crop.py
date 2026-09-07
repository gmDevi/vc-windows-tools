"""Download a crop of a public segment surface-volume zarr level (chunks [L,128,128], uncompressed uint8) into a local zarr-v2 array."""
import sys, os, json, urllib.request, urllib.parse, concurrent.futures as cf, time
B="https://vesuvius-challenge-open-data.s3.amazonaws.com/"
zpath, level = sys.argv[1], sys.argv[2]
y0, x0, ny, nx = map(int, sys.argv[3:7]); out = sys.argv[7]
za = json.loads(urllib.request.urlopen(B+urllib.parse.quote(f"{zpath}{level}/.zarray")).read().decode())
L, cy, cx = za['chunks']; assert y0 % cy == 0 and x0 % cx == 0
os.makedirs(out, exist_ok=True)
json.dump({**za, "shape": [za['shape'][0], ny, nx]}, open(os.path.join(out, '.zarray'), 'w'))
jobs = [(iy, ix) for iy in range(ny // cy) for ix in range(nx // cx)]
def get(j):
    iy, ix = j; dst = os.path.join(out, '0', str(iy), str(ix))
    if os.path.exists(dst): return 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = B + urllib.parse.quote(f"{zpath}{level}/0/{y0//cy+iy}/{x0//cx+ix}")
    for k in range(4):
        try:
            open(dst, 'wb').write(urllib.request.urlopen(url).read()); return 1
        except urllib.error.HTTPError as e:
            if e.code == 404: return 0
            time.sleep(1+k)
        except Exception: time.sleep(1+k)
    return 0
t0 = time.time()
with cf.ThreadPoolExecutor(24) as ex: got = sum(ex.map(get, jobs))
print(f"{got}/{len(jobs)} chunks, {time.time()-t0:.0f}s ->", out, 'layers', L)
