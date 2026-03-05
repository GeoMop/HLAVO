#!/usr/bin/env python3
# Trnasform wrong georeferencing of other quarter layers to the extent of 
# related files in Bohadlo_26_01_07. All the same transform, but differs from transform of the base.
#
# Work in CWD.
# For each Fin_*.tif:
#   - derive affine correction from ORIG/FIXED
#   - apply BLOWUP scale inside pixel->world transform (footprint scale)
#   - write <stem>.fix.tif alongside input
#
from pathlib import Path
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

# --- EDIT THESE ---
ORIG  = Path("_Fin_Q__Q-1-base_Q-1-base.tif")
FIXED = Path("_Orez-_Q-1-base.fix.tif")
BLOWUP = 2.0
# ------------------

def gt_to_M(gt):
    return np.array([[gt[1], gt[2], gt[0]],
                     [gt[4], gt[5], gt[3]],
                     [0.0,   0.0,   1.0]], dtype=float)

def M_to_gt(M):
    return (float(M[0,2]), float(M[0,0]), float(M[0,1]),
            float(M[1,2]), float(M[1,0]), float(M[1,1]))

# Derive affine A (wrong-world -> correct-world): A = Mf * inv(Mo)
ds_o = gdal.Open(str(ORIG),  gdal.GA_ReadOnly)
ds_f = gdal.Open(str(FIXED), gdal.GA_ReadOnly)
Mo = gt_to_M(ds_o.GetGeoTransform())
Mf = gt_to_M(ds_f.GetGeoTransform())
A  = Mf @ np.linalg.inv(Mo)
proj = ds_f.GetProjection()

print("Affine A (wrong-world -> correct-world):")
print(A)

# Pixel-axis scale matrix (applied in pixel space)
D = np.array([[BLOWUP, 0.0,    0.0],
              [0.0,    BLOWUP, 0.0],
              [0.0,    0.0,    1.0]], dtype=float)

for in_path in sorted(Path.cwd().glob("Fin_*.tif")):
    out_path = in_path.with_name(f"{in_path.stem}.fix.tif")

    ds = gdal.Open(str(in_path), gdal.GA_ReadOnly)

    # Scale pixel->world by BLOWUP (footprint), then map wrong-world->correct-world via A
    M  = gt_to_M(ds.GetGeoTransform())
    M2 = A @ (M @ D)
    gt2 = M_to_gt(M2)

    drv = gdal.GetDriverByName("GTiff")
    out = drv.CreateCopy(str(out_path), ds, strict=0)
    out.SetGeoTransform(gt2)
    out.SetProjection(proj)
    out.FlushCache()
    out = None

    print("WROTE:", out_path)
