**Plan**
- *Done.* Locate project structure under `/workspace` and confirm where to place code (likely `deep_model/gpt_zone`), plus the read-only GIS directory and `uhelna_all.qgz`.
- *Introduced.* Define YAML config schema (meshsteps, model dir, conductivities, modflow settings).
- *Done.* Implement GIS readers:
  - Load `JB_extended_domain` polygon.
  - Load `HG model layers` rasters in surface->bottom order.

- *Done.* Establish local coordinate origin (first polygon point rounded to 1000 m) and transform all XY to local coords.
- *Done.* Build grid:
  - Rectangular XY grid covering domain extent with meshsteps.
  - Active mask = inside/intersect polygon.
  - Vertical range from rasters; build Z levels.
  - create minimalistic pytest unit tests for previous stes and make it run
  - REVIEW

  - Assign materials bottom-up using `last_top` per column; 
  - refine Z to match layer boundaries and even spacing within each layer.
  - minimalistic unit test
  
- Write model inputs (geometry, materials, properties) into `model/` only.

- REVIEW

- Implement the model run script:
  - call builder
  - set no-flow sides/bottom
  - seepage face + 0.1 mm/day recharge on top
  - set conductivities from YAML
  - run steady Darcy flow
- Add a single entrypoint script to run build+model; verify end-to-end.
