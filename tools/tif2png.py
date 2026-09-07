"""Convert a (Big)TIFF ink prediction to a downsampled PNG for viewing, plus simple stats."""
import sys, numpy as np, tifffile
from PIL import Image
src=sys.argv[1]; dst=sys.argv[2]; ds=int(sys.argv[3]) if len(sys.argv)>3 else 2
a=tifffile.imread(src)
if a.ndim==3: a=a[0]
print('shape',a.shape,'dtype',a.dtype,'min',a.min(),'max',a.max(),'mean',round(float(a.mean()),2),
      'frac>128',round(float((a>128).mean()),4),'frac>200',round(float((a>200).mean()),4))
if ds>1:
    h,w=a.shape; a=a[:h//ds*ds,:w//ds*ds].reshape(h//ds,ds,w//ds,ds).max(axis=(1,3))
Image.fromarray(a.astype(np.uint8)).save(dst); print('wrote',dst,a.shape)
