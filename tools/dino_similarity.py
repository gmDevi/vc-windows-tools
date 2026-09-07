"""DINO ink-likeness: run the dinovol teacher backbone over a CT block in 128^3 windows and compute cosine similarity of every
8^3 patch token to the organizers' reference ink embedding (avg_ref_embedding.npy). Writes a similarity volume (uint8) + projections.
Usage: dino_similarity.py <ct.zarr> <backbone.pt> <avg_ref.npy> <out_dir> [win=128] [stride=96]"""
import sys, os, json, time, numpy as np, torch, zarr
from PIL import Image
sys.path.insert(0, 'C:/Users/mdevi/prize/vesuvius/dinovol_repo')
from dinovol_2.model.model import DinoVitStudentTeacher

ct_path, ckpt_path, ref_path, out = sys.argv[1:5]
win = int(sys.argv[5]) if len(sys.argv) > 5 else 128
stride = int(sys.argv[6]) if len(sys.argv) > 6 else 96
os.makedirs(out, exist_ok=True)
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = ck['config']['model'] if 'model' in ck['config'] else ck['config']
model = DinoVitStudentTeacher(cfg)
sd = ck['teacher']
res = model.teacher.load_state_dict(sd, strict=False)
print('loaded teacher; missing', len(res.missing_keys), 'unexpected', len(res.unexpected_keys), flush=True)
if res.missing_keys[:3]: print('  e.g. missing', res.missing_keys[:3])
backbone = model.teacher['backbone'].cuda().eval()
ref = torch.from_numpy(np.load(ref_path).astype(np.float32)).cuda(); ref = ref / ref.norm()

g = zarr.open(ct_path, mode='r'); ct = np.asarray(g['0'] if hasattr(g, 'keys') and '0' in g else g)
Z, Y, X = ct.shape; ps = 8
# checkpoint normalization scheme: look in config for 'normalization'; default to per-window z-score with clipping
norm = ck['config'].get('normalization', ck['config'].get('dataset', {}).get('normalization', 'zscore')) if isinstance(ck['config'], dict) else 'zscore'
print('normalization', norm, flush=True)
sim = np.zeros((Z // ps, Y // ps, X // ps), np.float32); cnt = np.zeros_like(sim)
def positions(n):
    p = list(range(0, max(n - win, 0) + 1, stride))
    if p[-1] != n - win: p.append(n - win)
    return p
t0 = time.time(); n = 0
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for z in positions(Z):
        for y in positions(Y):
            batch, coords = [], []
            for x in positions(X):
                p = ct[z:z+win, y:y+win, x:x+win].astype(np.float32)
                if (p > 0).mean() < 0.05: continue
                m, s = p[p > 0].mean(), p[p > 0].std() + 1e-3
                p = np.clip((p - m) / s, -5, 5)
                batch.append(p); coords.append((z, y, x))
                if len(batch) == 4 or x == positions(X)[-1]:
                    t = torch.from_numpy(np.stack(batch))[:, None].cuda()
                    outp = backbone.forward_features(t, masks=None, view_kind='global')
                    tok = outp['x_norm_patchtokens'].float()                       # (B, N, 864)
                    tok = tok / tok.norm(dim=-1, keepdim=True)
                    cs = (tok @ ref).float().reshape(len(batch), win // ps, win // ps, win // ps).cpu().numpy()
                    for (zz, yy, xx), c in zip(coords, cs):
                        sl = (slice(zz // ps, zz // ps + win // ps), slice(yy // ps, yy // ps + win // ps), slice(xx // ps, xx // ps + win // ps))
                        sim[sl] += c; cnt[sl] += 1; n += 1
                    batch, coords = [], []
        print(f'z={z}: {n} windows, {time.time()-t0:.0f}s', flush=True)
sim = np.where(cnt > 0, sim / np.maximum(cnt, 1), 0)
print('cosine stats: mean %.3f p90 %.3f p99 %.3f max %.3f frac>0.5 %.5f' % (sim.mean(), np.percentile(sim, 90), np.percentile(sim, 99), sim.max(), (sim > 0.5).mean()), flush=True)
np.save(os.path.join(out, 'dino_cos.npy'), sim)
s8 = (np.clip(sim, 0, 1) * 255).astype(np.uint8)
for name, im in [('maxz', s8.max(0)), ('maxy', s8.max(1)), ('maxx', s8.max(2)), ('midz', s8[s8.shape[0] // 2])]:
    Image.fromarray(np.kron(im, np.ones((4, 4), np.uint8))).save(os.path.join(out, f'dino_{name}.png'))
json.dump({'mean': float(sim.mean()), 'p99': float(np.percentile(sim, 99)), 'max': float(sim.max()), 'frac>0.5': float((sim > 0.5).mean())}, open(os.path.join(out, 'summary.json'), 'w'))
print('wrote', out, flush=True)
