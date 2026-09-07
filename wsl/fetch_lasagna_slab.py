"""Copy a z-slab of the organizers' lasagna normal fields (nx, ny, grad_mag) into local OME-Zarr groups laid out as the
spiral fitter expects (lasagna_inputs/las_008_<name>.ome.zarr/<group>), keeping the full array shape so coordinates match."""
import sys, os, json, numpy as np, zarr, fsspec, numcodecs
scroll, vol, run, group = sys.argv[1:5]
z0, z1 = int(sys.argv[5]), int(sys.argv[6]); out_root = sys.argv[7]
B = f"https://vesuvius-challenge-open-data.s3.amazonaws.com/{scroll}/representations/predictions/lasagna/{vol}-lasagna-{run}/"
os.makedirs(out_root, exist_ok=True)
for name in ['nx', 'ny', 'grad_mag']:
    src = zarr.open(fsspec.get_mapper(B + f"{scroll}_{name}.ome.zarr/{group}"), mode='r')
    attrs = json.loads(fsspec.open(B + f"{scroll}_{name}.ome.zarr/.zattrs").open().read())
    dst_dir = os.path.join(out_root, f"las_008_{name}.ome.zarr")
    os.makedirs(dst_dir, exist_ok=True)
    json.dump({"zarr_format": 2}, open(os.path.join(dst_dir, '.zgroup'), 'w'))
    json.dump(attrs, open(os.path.join(dst_dir, '.zattrs'), 'w'))
    dst = zarr.open(os.path.join(dst_dir, group), mode='a', shape=src.shape, chunks=src.chunks, dtype=src.dtype,
                    compressor=numcodecs.Blosc(cname='zstd', clevel=3, shuffle=1))
    step = 64
    for z in range(z0, z1, step):
        zz = min(z1, z + step)
        dst[z:zz] = np.asarray(src[z:zz])
        print(name, z, zz, flush=True)
    print('done', name, src.shape, src.chunks, flush=True)
