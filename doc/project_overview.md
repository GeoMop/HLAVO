# HLAVO project overview

The HLAVO (HLAdiny VOd = water table) project aims to develop a
comprehensive system for modeling and predicting the water table using
detailed meteorological data and infiltration estimates derived from an
assimilated soil model.

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

## Project software components
- `data ingress`
- `soil model` (with evapo-transpoiration and Kalman filter assimilation)
- `deep vadose zone model`
- `prediction and calibration engine`
- `visualization dashboard`

![HLAVO_schema](HLAVO_schema.svg)


## Application to the Uhelná locality

