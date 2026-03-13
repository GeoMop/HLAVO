import numpy as np
from flopy.utils.binaryfile import HeadFile
from flopy.utils import BinaryHeader

headfile_in = "uhelna.hds"
strtfile_out = "strt_all.bin"

# Read last heads from previous run
hds = HeadFile(headfile_in)
times = hds.get_times()
h_last = hds.get_data(totim=times[-1])   # shape: (nlay, nrow, ncol)

nlay, nrow, ncol = h_last.shape

# Build a standard MODFLOW binary header (52 bytes)
header = BinaryHeader.create(
    bintype="head",
    precision="double",
    text="HEAD",
    nrow=nrow,
    ncol=ncol,
    ilay=nlay,   # for full-grid file this value is not very important
    pertim=1.0,
    totim=1.0,
    kstp=1,
    kper=1,
)

# Write header + full 3D array
with open(strtfile_out, "wb") as f:
    header.tofile(f)
    np.asarray(h_last, dtype=np.float64).tofile(f)

print(f"Wrote {strtfile_out}")
print(f"Shape = {h_last.shape}")
print(f"Expected data bytes = {h_last.size * 8}")
print(f"Expected total bytes = {52 + h_last.size * 8}")