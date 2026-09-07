"""Face-on sheet views: estimate dominant sheet normal in xy from the raw block, rotate volume so sheets are parallel to the yz plane,
then max-project ink probability over thin x-slabs -> images where letters (if any) would appear as glyphs. Also writes raw mean-slab images."""
import sys, os, numpy as np, zarr
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
block, probdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
ds = int(sys.argv[4]) if len(sys.argv) > 4 else 2          # downsample factor for speed
half = int(sys.argv[5]) if len(sys.argv) > 5 else 6        # slab half-width (in downsampled voxels)
os.makedirs(out, exist_ok=True)
raw = np.asarray(zarr.open(block, mode='r')['0'])
prob = np.asarray(zarr.open(os.path.join(probdir, 'ink_prob_u8.zarr'), mode='r'))
if ds > 1:
    raw = raw[:raw.shape[0]//ds*ds, :raw.shape[1]//ds*ds, :raw.shape[2]//ds*ds].reshape(raw.shape[0]//ds, ds, raw.shape[1]//ds, ds, raw.shape[2]//ds, ds).mean(axis=(1,3,5)).astype(np.float32)
    prob = prob[:prob.shape[0]//ds*ds, :prob.shape[1]//ds*ds, :prob.shape[2]//ds*ds].reshape(prob.shape[0]//ds, ds, prob.shape[1]//ds, ds, prob.shape[2]//ds, ds).max(axis=(1,3,5))
# dominant normal direction in xy from gradient covariance of a few z-slices
gy = gx = 0; cov = np.zeros((2,2))
for z in range(0, raw.shape[0], max(1, raw.shape[0]//8)):
    s = ndi.gaussian_filter(raw[z], 2)
    dy, dx = np.gradient(s)
    cov += np.array([[ (dy*dy).sum(), (dy*dx).sum()], [(dy*dx).sum(), (dx*dx).sum()]])
w, v = np.linalg.eigh(cov); n = v[:, np.argmax(w)]      # normal (dy, dx)
theta = np.degrees(np.arctan2(n[0], n[1]))               # angle of normal from +x axis
print('normal (dy,dx)=', n.round(3), 'theta=%.1f deg' % theta, flush=True)
# rotate about z so that normal -> +x : rotate by -theta in the (y,x) plane
def aniso(img):
    dy, dx = np.gradient(ndi.gaussian_filter(img, 1.5)); return np.abs(dx).mean() / (np.abs(dy).mean() + 1e-6)
mid = raw[raw.shape[0]//2]
cands = {a: aniso(ndi.rotate(mid, a, reshape=False, order=1)) for a in (theta, -theta, theta+90, -theta+90, theta-90, -theta-90)}
ang = max(cands, key=cands.get); print('rotation candidates (x-gradient dominance):', {round(k,1): round(v,2) for k,v in cands.items()}, '-> using', round(ang,1), flush=True)
rp = ndi.rotate(prob, ang, axes=(1, 2), reshape=False, order=0)
rr = ndi.rotate(raw, ang, axes=(1, 2), reshape=False, order=1)
Z, Y, X = rp.shape
# sanity: after rotation sheets should be vertical lines in the (y,x) mid slice
Image.fromarray(np.clip(rr[Z//2], 0, 255).astype(np.uint8)).save(os.path.join(out, 'rot_midz_raw.png'))
Image.fromarray(rp[Z//2]).save(os.path.join(out, 'rot_midz_prob.png'))
tiles = []
step = max(1, half)
xs = list(range(half, X - half, step))
for x in xs:
    pslab = rp[:, :, x-half:x+half+1].max(axis=2)          # (Z, Y) face-on ink
    rslab = rr[:, :, x-half:x+half+1].mean(axis=2)
    if (pslab > 128).mean() < 0.002: continue
    tiles.append((x, pslab, rslab))
print(len(tiles), 'slabs with ink candidates', flush=True)
cell = max(Z, Y)
cols = 4
rows = (len(tiles) + cols - 1) // cols
for kind in ['prob', 'raw']:
    sheet = Image.new('L', (cols*Y, rows*Z), 0)
    for i, (x, p, r) in enumerate(tiles):
        im = p if kind == 'prob' else np.clip(r, 0, 255).astype(np.uint8)
        t = Image.fromarray(im); d = ImageDraw.Draw(t); d.text((3, 3), f'x={x*ds}', fill=255)
        sheet.paste(t, ((i % cols)*Y, (i // cols)*Z))
    sheet.save(os.path.join(out, f'faceon_{kind}.png')); print('wrote', kind, sheet.size, flush=True)
