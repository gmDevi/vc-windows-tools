"""Summarize a 3D logits/softmax zarr from vesuvius inference: stats + max/mean projections + mid slices as PNG."""
import sys, os, glob, numpy as np, zarr
from PIL import Image
out_dir=sys.argv[1]; prefix=sys.argv[2]
paths=sorted(glob.glob(os.path.join(out_dir,'logits_part_*.zarr')))+sorted(glob.glob(os.path.join(out_dir,'*.zarr')))
p=paths[0]; g=zarr.open(p,mode='r')
arr=g if hasattr(g,'shape') else (g['0'] if '0' in g else g[list(g.array_keys())[0]])
a=np.asarray(arr); print('path',p,'shape',a.shape,'dtype',a.dtype)
if a.ndim==4: a=a[0] if a.shape[0]<=2 else a[...,0]
if a.dtype!=np.uint8:
    a=a.astype(np.float32)
    if a.min()<0 or a.max()>1.0001: a=1/(1+np.exp(-a))
    print('prob stats: mean %.4f p99 %.4f max %.4f frac>0.5 %.5f'%(a.mean(),np.percentile(a,99),a.max(),(a>0.5).mean()))
    a8=(a*255).astype(np.uint8)
else: a8=a; print('uint8 stats mean %.1f frac>128 %.5f'%(a.mean(),(a>128).mean()))
def save(im,name): Image.fromarray(im).save(os.path.join(out_dir,f'{prefix}_{name}.png'))
save(a8.max(0),'maxz'); save(a8.max(1),'maxy'); save(a8.max(2),'maxx')
z,y,x=[s//2 for s in a8.shape]; save(a8[z],'midz'); save(a8[:,y],'midy'); save(a8[:,:,x],'midx')
print('wrote projections to',out_dir)
