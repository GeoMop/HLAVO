import flopy
from pathlib import Path
import numpy as np
from flopy.utils.binaryfile import HeadFile
from flopy.utils import BinaryHeader


# Folder that contains mfsim.nam
sim_ws = Path(".")
headfile_in = "uhelna.hds"
strtfile_out = "strt_all.bin"
nsteps = 2
nwells = 2
wells = [ (68, 34), (68, 102) ]
w0 =  [ 1.0, 1.1, 1.0, 0.8, 0.5, 0.8, 0.9, 0.8, 0.7, 0.9 ]
w1 =  [ 0.5, 0.6, 0.6, 0.4, 0.3, 0.2, 0.3, 0.4, 0.3, 0.5 ]
nr = 165
nc = 137

def create_recharge( step ):
    rch = np.zeros((nr, nc), dtype=float)
    # split at half of y direction = half of rows
    half_row = nr // 2

    rch[:half_row, :] = w0[ step ] * 1e-4
    rch[half_row:, :] = w1[ step ] * 1e-4
    return rch

sim = flopy.mf6.MFSimulation.load(
    sim_ws=str(sim_ws),
    verbosity_level=1,
)

for i in range( nsteps ):
    print( f"This is composed model step no. {i}" )
    print( f"From 1D models got recharges:" )
    print( f"Well 0: {w0[ i ]}")
    print( f"Well 1: {w1[ i ]}")
# Load existing MF6 simulation

    recharge=create_recharge( i )
    np.savetxt("recharge.txt", recharge, fmt="%.8e")
    
    # Run it
    success, buff = sim.run_simulation()

    print("Success:", success)
    if not success:
        print("\n".join(buff))
        raise RuntimeError("MODFLOW 6 did not terminate normally.")


    # Read last heads from previous run
    hds = HeadFile(headfile_in)
    times = hds.get_times()
    h_last = hds.get_data(totim=times[-1])   # shape: (nlay, nrow, ncol)

    nlay, nrow, ncol = h_last.shape
    # print( f"nrow = {nrow}, ncol = {ncol}")

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

    # print(f"Wrote {strtfile_out}")
    # print(f"Shape = {h_last.shape}")
    # print(f"Expected data bytes = {h_last.size * 8}")
    # print(f"Expected total bytes = {52 + h_last.size * 8}")

    # print("Heads on wells cells:")
    for i, w in enumerate( wells) :
        # print( f"Well {i}, location {w}, head: {h_last[ 0, w[ 0 ], w[ 1 ] ]}")
        print( f"Sending to 1D model #{i} head {h_last[ 0, w[ 0 ], w[ 1 ] ]}")