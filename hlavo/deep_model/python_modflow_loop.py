import numpy as np
import flopy
from flopy.utils.binaryfile import HeadFile
from flopy.utils import BinaryHeader, Util2d

# 1) Read last time in the head file from yesterday's run
hds = HeadFile("uhelna.hds")  # your head save file
times = hds.get_times()
h_last = hds.get_data(totim=times[-1])   # shape (nlay, nrow, ncol) for DIS

nlay, nrow, ncol = h_last.shape
precision = "double"  # MF6 binary array input is double precision in the IO spec  [oai_citation:2‡USGS](https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.5.0.pdf)

# 2) Write MF6-readable binary array input, one file per layer
for ilay in range(nlay):
    header = BinaryHeader.create(
        bintype="head",
        precision=precision,
        text="head",
        nrow=nrow,
        ncol=ncol,
        ilay=ilay + 1,
        pertim=1.0, totim=1.0, kstp=1, kper=1
    )
    Util2d.write_bin((nrow, ncol), f"strt.layer{ilay+1}.bin", h_last[ilay], header_data=header)

# 3) In your next run’s IC package, point STRT to these files:
# STRT LAYERED
#   OPEN/CLOSE strt.layer1.bin (BINARY)
#   OPEN/CLOSE strt.layer2.bin (BINARY)
#   ...