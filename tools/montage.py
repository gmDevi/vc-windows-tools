"""Build a contact sheet PNG from all prediction TIFFs in preds/ (forward+reverse side by side), plus per-file stats."""
import sys, glob, os, numpy as np, tifffile
from PIL import Image, ImageDraw
root=sys.argv[1]; out=sys.argv[2]; cell=int(sys.argv[3]) if len(sys.argv)>3 else 640
files=sorted(f for f in glob.glob(os.path.join(root,'*.tif')) if not f.endswith('_reverse.tif'))
tiles=[]
for f in files:
    for suf in ['', '_reverse']:
        p=f[:-4]+suf+'.tif'
        if not os.path.exists(p): continue
        a=tifffile.imread(p); a=a[0] if a.ndim==3 else a
        h,w=a.shape; s=max(1,int(np.ceil(max(h,w)/cell)))
        b=a[:h//s*s,:w//s*s].reshape(h//s,s,w//s,s).max(axis=(1,3)) if s>1 else a
        im=Image.fromarray(b.astype(np.uint8)).convert('L')
        canvas=Image.new('L',(cell,cell),0); canvas.paste(im,(0,0))
        d=ImageDraw.Draw(canvas); d.text((4,4),f"{os.path.basename(p)[:28]} mean={a.mean():.0f} >128:{(a>128).mean()*100:.1f}%",fill=255)
        tiles.append(canvas)
        print(f"{os.path.basename(p):40s} shape={a.shape} mean={a.mean():.1f} p99={np.percentile(a,99):.0f} frac>128={(a>128).mean():.4f} frac>180={(a>180).mean():.5f}")
cols=4; rows=(len(tiles)+cols-1)//cols
sheet=Image.new('L',(cols*cell,rows*cell),0)
for i,t in enumerate(tiles): sheet.paste(t,((i%cols)*cell,(i//cols)*cell))
sheet.save(out); print('wrote',out,sheet.size)
