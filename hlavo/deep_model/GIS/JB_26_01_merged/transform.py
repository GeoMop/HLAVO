#!/usr/bin/env python3
# Transform original Qbase layer with wrong georeferencing and smaller pixel size to 
# extend of SX-merged.
#
# Edit these two to point at ONE representative pair:
# - ORIG:   the raster BEFORE your manual gdal_edit fix (wrong georef)
# - FIXED:  the SAME raster AFTER your manual gdal_edit fix (correct georef)
ORIG  = "_Fin__Base_Q_Surface_final_29_7.tif"
FIXED = "_Fin__Base_Q_Surface_final_29_7.fix.tif"

import glob
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

def gt_to_M(gt):
    return np.array([[gt[1], gt[2], gt[0]],
                     [gt[4], gt[5], gt[3]],
                     [0.0,   0.0,   1.0]], dtype=float)

def M_to_gt(M):
    return (float(M[0,2]), float(M[0,0]), float(M[0,1]),
            float(M[1,2]), float(M[1,0]), float(M[1,1]))

# derive world->world affine: A = Mf * inv(Mo)
ds_o = gdal.Open(ORIG,  gdal.GA_ReadOnly)
ds_f = gdal.Open(FIXED, gdal.GA_ReadOnly)
Mo = gt_to_M(ds_o.GetGeoTransform())
Mf = gt_to_M(ds_f.GetGeoTransform())
A  = Mf @ np.linalg.inv(Mo)
proj = ds_f.GetProjection()

print("Affine A (wrong-world -> correct-world):")
print(A)

# apply to all Fin_*.tif in cwd (in place)
for path in sorted(glob.glob("Fin_*.tif")):
    ds = gdal.Open(path, gdal.GA_Update)
    M  = gt_to_M(ds.GetGeoTransform())
    M2 = A @ M
    ds.SetGeoTransform(M_to_gt(M2))
    ds.SetProjection(proj)
    ds.FlushCache()
    ds = None
    print("UPDATED:", path)
