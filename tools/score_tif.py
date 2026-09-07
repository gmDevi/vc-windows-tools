"""Score an ink-prediction TIFF: mean, fraction >200, count of letter-sized bright blobs (400..30000 px at ~9um), p99.9. Prints JSON."""
import sys, json, numpy as np, tifffile
from scipy import ndimage as ndi
a = tifffile.imread(sys.argv[1]); a = a[0] if a.ndim == 3 else a
hi = a > 200; lab, n = ndi.label(hi)
sizes = ndi.sum(hi, lab, range(1, n + 1)) if n else np.array([])
print(json.dumps({'mean': float(a.mean()), 'frac200': float(hi.mean()),
                  'blobs_letterlike': int(((sizes > 400) & (sizes < 30000)).sum()),
                  'p999': float(np.percentile(a, 99.9))}))
