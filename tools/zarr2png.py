"""Render layers of an uncompressed zarr-v2 surface volume (chunks [Z,128,128], '/' separator) to PNG without zarr installed."""
import json, os, sys, numpy as np
from PIL import Image
def load_level(zdir, level='1'):
    meta=json.load(open(os.path.join(zdir,level,'.zarray')))
    Z,Y,X=meta['shape']; cz,cy,cx=meta['chunks']
    assert meta['compressor'] is None and meta['dtype']=='|u1'
    arr=np.zeros((Z,Y,X),np.uint8)
    for iy in range((Y+cy-1)//cy):
        for ix in range((X+cx-1)//cx):
            p=os.path.join(zdir,level,'0',str(iy),str(ix))
            if not os.path.exists(p): continue
            buf=np.frombuffer(open(p,'rb').read(),np.uint8)
            if buf.size!=cz*cy*cx: continue
            blk=buf.reshape(cz,cy,cx)
            y1=min(Y,(iy+1)*cy); x1=min(X,(ix+1)*cx)
            arr[:,iy*cy:y1,ix*cx:x1]=blk[:, :y1-iy*cy, :x1-ix*cx]
    return arr
if __name__=='__main__':
    zdir=sys.argv[1]; out=sys.argv[2]; level=sys.argv[3] if len(sys.argv)>3 else '1'
    a=load_level(zdir,level)
    z=a.shape[0]//2
    Image.fromarray(a[z]).save(out)
    Image.fromarray(a.max(axis=0)).save(out.replace('.png','_max.png'))
    print('shape',a.shape,'nonzero frac',round(float((a[z]>0).mean()),3),'mean',round(float(a[z].mean()),1))
