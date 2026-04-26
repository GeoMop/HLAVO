from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path

import attrs
import numpy as np
import xarray


from hlavo.ingress.moist_profile.load_zarr_data import load_measurments_data, load_meteo_data
from hlavo.kalman.kalman import KalmanFilter
from hlavo.misc import resolve_named_class
from hlavo.misc.config import load_config

LOG = logging.getLogger(__name__)


def dataset_time_slice(dataset, start_time: np.datetime64, stop_time: np.datetime64):
    return dataset.sel(date_time=slice(start_time, stop_time))


@attrs.define(frozen=True)
class Model1DLocation:
    idx: int
    longitude: float
    latitude: float


@attrs.define
class Model1DData:
    long_lat: float
    profiles_dataset: xarray.Dataset
    surface_dataset: xarray.Dataset

    @classmethod
    def from_config(cls, site_id, composed:'ComposedData', schemas) -> "Model1DData":
        select = lambda ds: ds.sel(site_id=site_id, date_time=slice(composed.start, composed.end)).compute()
        profiles = select(load_measurments_data(scheme_file=
                                         composed.relative_resolve(schemas['profiles'])))
        LOG.debug("Loaded 1D profile dataset for site_id=%s: %s", site_id, profiles)
        start_profiles =  profiles.isel(date_time=0)
        long_lat = (start_profiles["longitude"].item(), start_profiles["latitude"].item())
        surface = select(load_meteo_data(scheme_file=
                                  composed.relative_resolve(schemas['surface'])))
        LOG.debug("Loaded 1D surface dataset for site_id=%s: %s", site_id, surface)

        return cls(
            long_lat,
            profiles,
            surface,
        )

    @property
    def longitude(self):
        return self.long_lat[0]

    @property
    def latitude(self):
        return self.long_lat[1]


@attrs.define
class KalmanMock:
    fixed_velocity: float = 0.1
    longitude: float = 14.889853
    latitude: float = 50.863565

    @classmethod
    def from_config(cls, workdir, config_source, verbose=False, seed=None):
        config_data, _ = load_config(config_source)
        model_1d_cfg = config_data.get("model_1d", config_data)
        assert isinstance(model_1d_cfg, dict), "model_1d config must be a mapping"
        _ = workdir
        _ = verbose
        _ = seed
        return cls(fixed_velocity=float(model_1d_cfg.get("mock_velocity", 0.0)))

    def kalman_step(self, ukf, measurements, meteo, pressure_at_bottom) -> float:
        _ = ukf
        _ = measurements
        _ = meteo
        _ = pressure_at_bottom
        return self.fixed_velocity

    def set_kalman_filter(self, kalman_R_matrix):
        return kalman_R_matrix

    def save_results(self):
        return None

@attrs.define
class Model1D:
    composed: 'ComposedData'
    site_id: int
    moisture_sigma: float
    data: Model1DData
    kalman: KalmanFilter | KalmanMock

    @classmethod
    def from_config(cls, composed, site_id: int, config: dict) -> "Model1D":
        data = Model1DData.from_config(site_id, composed, config['schema_files'])

        kalman_class = resolve_named_class(config['kalman_class_name'], (KalmanFilter, KalmanMock))
        mcfg = config.get('model_config', {})
        clm_f = mcfg.get('clm_files', {})
        mcfg['clm_files'] = {k: composed.relative_resolve(v) for k, v in clm_f.items()}
        kalman = kalman_class.from_config(
            composed.workdir,
            config,
            verbose=False,
            seed=composed.seed,
        )
        return Model1D(
                composed=composed,
                site_id=site_id,
                moisture_sigma=float(config["moisture_sigma"]),
                data=data,
                kalman=kalman)

    @property
    def longitude(self):
        return self.data.longitude

    @property
    def latitude(self):
        return self.data.latitude

    @cached_property
    def ukf(self):
        return self.kalman.set_kalman_filter(self.R_matrix)

    @cached_property
    def R_matrix(self):
        n_sensors_per_probe = len(self.data.profiles_dataset["depth_level"])
        return self.moisture_sigma * np.eye(n_sensors_per_probe)


    def _get_meteo_data(self):
        meteo_dataset = load_meteo_data(self.kalman.measurements_config["meteo_scheme_file"])
        self.meteo_dataset = meteo_dataset.sel(site_id=self.site_id)




    def step(self, start_time, target_time, pressure_at_bottom):
        measurements = dataset_time_slice(self.data.profiles_dataset, start_time, target_time)
        meteo = dataset_time_slice(self.data.surface_dataset, start_time, target_time)

        darcy_velocity = None
        if len(measurements) > 0:
            darcy_velocity = self.kalman.kalman_step(
                self.ukf,
                measurements,
                meteo,
                pressure_at_bottom,
            )
        # TODO: more detailed output and either send through Queue to 3D worker and
        # save to ZARR from there, or excersize zarr parallel write (preallocation and suitable chunking necessary)
        return darcy_velocity

    def save_results(self):
        self.kalman.save_results()

