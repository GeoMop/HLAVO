"""Toy MF6 geometry injected into Model3DAPI by the composed tests.

A small unconfined box: 3 layers, water table initially at -60 m, a constant head
column (-60 m) on the east side draining the recharge. The west half of the columns
belongs to 1D site 0, the east half to site 1. One inactive bottom cell exercises
the reduced-node (IDOMAIN) mapping.
"""
from __future__ import annotations

import flopy
import numpy as np

from hlavo.composed.model_3d_api import NO_SITE, GwfSetup


def build_cube(sim: flopy.mf6.MFSimulation, locations_1d: list[int], config: dict) -> GwfSetup:
    nlay, nrow, ncol = 3, int(config.get("nrow", 2)), int(config.get("ncol", 4))
    initial_head = float(config.get("initial_head", -60.0))
    gwf = flopy.mf6.ModflowGwf(sim, modelname="cube", save_flows=False)

    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    idomain[-1, 0, 0] = 0
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=nlay, nrow=nrow, ncol=ncol, delr=100.0, delc=100.0,
        top=0.0, botm=[-70.0, -85.0, -100.0], idomain=idomain,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=initial_head)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=[1, 0, 0], k=float(config.get("k", 1.0)))
    flopy.mf6.ModflowGwfsto(gwf, iconvert=[1, 0, 0], sy=0.1, ss=1.0e-5, transient={0: True})
    east = ncol - 1
    flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(lay, row, east), initial_head] for lay in range(nlay) for row in range(nrow)]
    )
    flopy.mf6.ModflowGwfrcha(gwf, pname="rcha", recharge=0.0)

    assert len(locations_1d) == 2, "cube test geometry is laid out for two 1D sites"
    cell_owner = np.full((nrow, ncol), NO_SITE, dtype=int)
    cell_owner[:, : ncol // 2] = locations_1d[0]
    cell_owner[:, ncol // 2 :] = locations_1d[1]
    return GwfSetup(gwf=gwf, recharge_package="rcha", cell_owner=cell_owner)
