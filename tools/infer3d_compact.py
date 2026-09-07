"""Compact full-3D ink inference: sliding 256^3 window over a local zarr block, blended in memory, uint8 zarr+PNG projections out.
Reuses the vesuvius train.py checkpoint loader (NetworkFromConfig) exactly like vesuvius.models.run.inference but without per-patch logits on disk."""
import sys, os, json, time, numpy as np, torch, zarr
from PIL import Image
sys.path.insert(0, 'C:/Users/mdevi/prize/vesuvius/villa/vesuvius/src')
from vesuvius.models.run import inference as VI

block, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
ps = int(sys.argv[4]) if len(sys.argv) > 4 else 256
stride = int(sys.argv[5]) if len(sys.argv) > 5 else 192
os.makedirs(out, exist_ok=True)

# --- build model the same way the package does (percentile_minmax normalization) ---
inf = VI.VesuviusInferer.__new__(VI.VesuviusInferer) if hasattr(VI, 'VesuviusInferer') else None
ckdata = torch.load(ckpt, map_location='cpu', weights_only=False)
model_config = VI._normalize_train_py_model_config(ckdata)
class Mgr:
    def __init__(s, mc):
        s.model_config = mc; s.targets = mc.get('targets', {})
        s.train_patch_size = mc.get('train_patch_size', mc.get('patch_size', (128,128,128)))
        s.train_batch_size = mc.get('train_batch_size', mc.get('batch_size', 2))
        s.in_channels = mc.get('in_channels', 1); s.autoconfigure = mc.get('autoconfigure', False)
        s.enable_deep_supervision = bool(mc.get('enable_deep_supervision', False)); s.model_name = mc.get('model_name','Model')
        s.spacing = [1]*len(s.train_patch_size)
model = VI.NetworkFromConfig(Mgr(model_config)).cuda().eval()
sd, src = VI._select_train_py_state_dict(ckdata)
sd = {k.replace('module.','').replace('_orig_mod.',''): v for k, v in sd.items()}
model.load_state_dict(sd, strict=True); print('loaded', src, 'targets', list(model_config.get('targets', {}).keys()), flush=True)
norm = VI._checkpoint_normalization_scheme(ckdata); print('normalization', norm, flush=True)

vol = np.asarray(zarr.open(block, mode='r')['0']); Z, Y, X = vol.shape; print('block', vol.shape, flush=True)
def positions(n): 
    p = list(range(0, max(n - ps, 0) + 1, stride))
    if p[-1] != n - ps: p.append(n - ps)
    return p
acc = np.zeros(vol.shape, np.float32); wsum = np.zeros(vol.shape, np.float32)
# gaussian blend window
g1 = np.exp(-0.5 * ((np.arange(ps) - (ps-1)/2) / (ps/8)) ** 2).astype(np.float32)
w = (g1[:,None,None] * g1[None,:,None] * g1[None,None,:]); w = np.maximum(w, 1e-3)
t0 = time.time(); n = 0
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for z in positions(Z):
        for y in positions(Y):
            for x in positions(X):
                p = vol[z:z+ps, y:y+ps, x:x+ps].astype(np.float32)
                if norm == 'percentile_minmax':
                    lo, hi = np.percentile(p, 0.5), np.percentile(p, 99.5)
                    p = np.clip((p - lo) / max(hi - lo, 1e-6), 0, 1)
                else:
                    p = (p - p.mean()) / (p.std() + 1e-6)
                t = torch.from_numpy(p)[None, None].cuda()
                o = model(t)
                if isinstance(o, dict): o = o.get('ink', next(iter(o.values())))
                if isinstance(o, (list, tuple)): o = o[0]
                prob = torch.sigmoid(o.float())[0, 0].cpu().numpy()
                acc[z:z+ps, y:y+ps, x:x+ps] += prob * w; wsum[z:z+ps, y:y+ps, x:x+ps] += w; n += 1
        print(f'z={z} done, {n} patches, {time.time()-t0:.0f}s', flush=True)
prob = acc / np.maximum(wsum, 1e-6); del acc, wsum
p8 = (prob * 255).astype(np.uint8)
print('prob stats: mean %.4f p99 %.4f p99.9 %.4f max %.4f frac>0.5 %.5f frac>0.8 %.6f' % (prob.mean(), np.percentile(prob,99), np.percentile(prob,99.9), prob.max(), (prob>0.5).mean(), (prob>0.8).mean()), flush=True)
zarr.save(os.path.join(out, 'ink_prob_u8.zarr'), p8)
for name, im in [('maxz', p8.max(0)), ('maxy', p8.max(1)), ('maxx', p8.max(2)), ('midz', p8[Z//2]), ('midy', p8[:, Y//2]), ('midx', p8[:, :, X//2])]:
    Image.fromarray(im).save(os.path.join(out, f'{name}.png'))
json.dump({'block': block, 'ckpt': ckpt, 'patch': ps, 'stride': stride, 'mean': float(prob.mean()), 'frac>0.5': float((prob>0.5).mean())}, open(os.path.join(out, 'summary.json'), 'w'))
print('wrote', out, flush=True)
