"""Run the canonical 2um ink model (ResNet3D-152 + 3D decoder, scrollprize/ink_canonical_2um) on a local sheet stack.
Usage: infer_canonical.py <sheet.zarr (group with '0' of shape (C,H,W) or (H,W,C))> <ckpt> <out_png_prefix> [stride=64] [reverse=0]
"""
import sys, os, numpy as np, zarr, torch
sys.path.insert(0, 'C:/Users/mdevi/prize/vesuvius/villa/ink-detection/optimized_inference')
import inference as INF
from model_resnet3d_3d_decoder import load_model
from PIL import Image

src, ckpt, out = sys.argv[1:4]
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 64
reverse = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
g = zarr.open(src, mode='r'); a = np.asarray(g['0'] if hasattr(g, 'keys') and '0' in g else g)
if a.shape[0] == min(a.shape):           # (C,H,W) -> (H,W,C)
    a = np.transpose(a, (1, 2, 0))
H, W, C = a.shape; print('stack', a.shape, flush=True)
CFG = INF.CFG
CFG.model_type = 'resnet3d-152-3d-decoder'; CFG.in_chans = C; CFG.size = 256; CFG.tile_size = 256; CFG.stride = stride
CFG.batch_size = 4; CFG.workers = 0; CFG.prefetch_factor = None
CFG.use_zarr_compression = False; CFG.zarr_output_dir = os.path.dirname(out) + '/_partitions'
os.makedirs(CFG.zarr_output_dir, exist_ok=True)
device = torch.device('cuda')
model = load_model(ckpt, device, num_frames=C)
res = INF.run_inference(np.ascontiguousarray(a), model, device, is_reverse_segment=reverse)
pred = np.asarray(zarr.open(res['mask_pred'], mode='r')); cnt = np.asarray(zarr.open(res['mask_count'], mode='r'))
prob = np.where(cnt > 0, pred / np.maximum(cnt, 1e-6), 0.0)
print('pred shape', prob.shape, 'mean %.4f p99 %.4f max %.4f frac>0.5 %.5f' % (prob.mean(), np.percentile(prob, 99), prob.max(), (prob > 0.5).mean()), flush=True)
p8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
Image.fromarray(p8).save(out + ('_rev' if reverse else '') + '.png')
np.save(out + ('_rev' if reverse else '') + '.npy', prob.astype(np.float16))
print('wrote', out, flush=True)
