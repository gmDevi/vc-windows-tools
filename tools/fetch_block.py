"""Download an axis-aligned block of a public OME-Zarr level-0 volume (uncompressed uint8, 128^3 chunks) into a local zarr-v2 store."""
import sys, os, json, urllib.request, urllib.parse, concurrent.futures as cf, time, numpy as np
B="https://vesuvius-challenge-open-data.s3.amazonaws.com/"
zpath=sys.argv[1]            # e.g. PHerc1203/volumes/....zarr/
z0,y0,x0=map(int,sys.argv[2:5]); n=int(sys.argv[5]); out=sys.argv[6]
c=128; assert z0%c==0 and y0%c==0 and x0%c==0 and n%c==0
level=sys.argv[7] if len(sys.argv)>7 else '0'
arr=os.path.join(out,'0'); os.makedirs(arr,exist_ok=True)
json.dump({"zarr_format":2,"shape":[n,n,n],"chunks":[c,c,c],"dtype":"|u1","compressor":None,"fill_value":0,"order":"C","filters":None,"dimension_separator":"/"},open(os.path.join(arr,'.zarray'),'w'))
json.dump({"zarr_format":2},open(os.path.join(out,'.zgroup'),'w'))
json.dump({"multiscales":[{"axes":[{"name":a,"type":"space","unit":"micrometer"} for a in "zyx"],"datasets":[{"path":"0","coordinateTransformations":[{"type":"scale","scale":[1,1,1]}]}],"version":"0.4","name":"block"}],"source":zpath,"origin_zyx":[z0,y0,x0],"level":level},open(os.path.join(out,'.zattrs'),'w'))
jobs=[(iz,iy,ix) for iz in range(n//c) for iy in range(n//c) for ix in range(n//c)]
def get(j):
    iz,iy,ix=j; dst=os.path.join(arr,str(iz),str(iy),str(ix))
    if os.path.exists(dst) and os.path.getsize(dst)==c**3: return 1
    os.makedirs(os.path.dirname(dst),exist_ok=True)
    url=B+urllib.parse.quote(f"{zpath}{level}/{z0//c+iz}/{y0//c+iy}/{x0//c+ix}")
    for k in range(4):
        try:
            buf=urllib.request.urlopen(url).read()
            if len(buf)!=c**3: return 0
            open(dst,'wb').write(buf); return 1
        except urllib.error.HTTPError as e:
            if e.code==404: return 0   # missing chunk == all zeros (masked)
            time.sleep(1+k)
        except Exception: time.sleep(1+k)
    return 0
t0=time.time()
with cf.ThreadPoolExecutor(24) as ex: got=sum(ex.map(get,jobs))
print(f"{got}/{len(jobs)} chunks present, {time.time()-t0:.0f}s ->",out)
