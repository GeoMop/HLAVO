#!/bin/bash

work_dir=$1
HLAVODIR=$2


docker pull flow123d/hlavo:0.1.0

docker run -it --name hlavo_tmp \
  flow123d/hlavo:0.1.0 \
  python3 -m pip install joblib

docker commit hlavo_tmp flow123d/hlavo:0.1.0
docker rm hlavo_tmp

docker run --rm \
  --user $(id -u):$(id -g) \
  -v ${HLAVODIR}:${HLAVODIR} \
  -v ${work_dir}:${work_dir} \
  -e PYTHONPATH=${HLAVODIR} \
  flow123d/hlavo:0.1.0 \
  python3 -m hlavo.kalman.kalman \
  ${work_dir} \
  ${HLAVODIR}/runs/kalman/config_test.yaml
