# Introduction

The HLAVO (HLAdiny VOd = water table) project develops a system for
predicting groundwater table dynamics using meteorological inputs and
assimilated soil moisture profile measurements, combining surface and deep
vadose zone modeling.

## Installation
<TODO: describe here, how to get the 0.1.0 tag from github, extract, run dev/hlavo-build pull>

### Environments
- Dev environment in `dev/` (Docker/conda wrappers and build scripts).
- Deep model sandboxing and QGIS-related tooling in `hlavo/deep_model/`.

## Top-level scripts (planned)
- `hlavo/kalman/kalman.py`: surface model assimilation runs.
- `hlavo/deep_model/deep.py`: deep model build + MODFLOW run.
- `hlavo/composed/main.py`: parallel coupling of surface + deep models.

## Locality: Uhelná

The primary test site for the HLAVO project is Uhelná, a small settlement
on the border with Poland. The key water source in Uhelná shows a
long-term declining trend in water table levels, potentially influenced by
nearby excavations: a sand mine (approximately 40 m deep) in the vicinity
and a brown coal surface mine (approximately 300 m deep) farther to the
north.

Due to the ongoing Czech–Polish “Turow dispute” an extensive geological
survey is underway in the area, including continuous water table
monitoring in many boreholes. Because the water catchment area is very
small, the locality is well suited for developing a detailed infiltration
and water table model.

The goal is to enable long-term prediction of water table evolution and to
support modeling of remediation measures at this locality and other similar
sites in the Bohemian Cretaceous Basin (česká křídová pánev) region.
