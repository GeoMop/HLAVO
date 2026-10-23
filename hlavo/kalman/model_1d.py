from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path

import attrs
import numpy as np
import xarray
from zarr_fuse.units import ureg


from hlavo.ingress.moist_profile.load_zarr_data import load_measurments_data, load_meteo_data
from hlavo.kalman.kalman import KalmanFilter
from hlavo.misc import resolve_named_class
from hlavo.misc.config import load_config

LOG = logging.getLogger(__name__)


def dataset_time_slice(dataset, start_time: np.datetime64, stop_time: np.datetime64):
    """Select the half-open window [start_time, stop_time).

    Label slicing in xarray includes both ends, which would feed a sample lying
    exactly on a step boundary into two consecutive steps.
    """
    date_time = dataset["date_time"].values
    mask = (date_time >= np.datetime64(start_time)) & (date_time < np.datetime64(stop_time))
    return dataset.isel(date_time=np.flatnonzero(mask))


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
    def load_sites(cls, site_ids: list[int], composed: 'ComposedData', config: dict) -> dict[int, "Model1DData"]:
        """Load the inputs of all 1D sites: site_id -> Model1DData.

        Opens each input store (config `schema_files`: profiles, surface) once and selects the
        data of every site. Called in the main process before the 1D workers start; the workers
        get the selected data and never open a store, because `zarr_fuse.open_store` writes group
        metadata on every open and concurrent opens collide (doc/question_zarr_fuse_open_store.md).
        """
        schemas = config["schema_files"]
        profiles_all = load_measurments_data(scheme_file=composed.relative_resolve(schemas['profiles']))
        surface_all = load_meteo_data(scheme_file=composed.relative_resolve(schemas['surface']))
        return {
            int(site_id): cls.from_datasets(int(site_id), composed, profiles_all, surface_all)
            for site_id in site_ids
        }

    @classmethod
    def from_datasets(cls, site_id: int, composed: 'ComposedData', profiles_all: xarray.Dataset,
                      surface_all: xarray.Dataset) -> "Model1DData":
        """Data of one site: the simulated interval of the site, loaded into memory.

        profiles_all, surface_all: datasets of all sites (lazy, as opened from the stores).
        """
        # Load with the local synchronous scheduler, not through the Dask cluster of the run
        # (a .compute() inside a worker task would wait for a free worker thread and deadlock).
        select = lambda ds: ds.sel(site_id=site_id, date_time=slice(composed.start, composed.end)).compute(
            scheduler="synchronous"
        )
        profiles = select(profiles_all)
        LOG.debug("Loaded 1D profile dataset for site_id=%s: %s", site_id, profiles)
        start_profiles = profiles.isel(date_time=0)
        long_lat = (start_profiles["longitude"].item(), start_profiles["latitude"].item())
        surface = select(surface_all)
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
class SurfaceMock:
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

    def kalman_step(self, ukf, measurements, meteo, pressure_at_bottom, site_id=None) -> float:
        _ = ukf
        _ = measurements
        _ = meteo
        _ = pressure_at_bottom
        _ = site_id
        return self.fixed_velocity

    def set_kalman_filter(self, kalman_R_matrix):
        return kalman_R_matrix

    def save_results(self):
        return None


@attrs.define
class SurfaceScalingMock:
    """Mock surface model: recharge [m/day] = scaling(mean profile moisture) * mean precipitation.

    The precipitation (variable `precipitation_var` of the meteo dataset) is averaged over the last
    48 h of the step and converted to m/day from its declared unit (variable attribute `unit`).
    """
    longitude: float = 14.889853
    latitude: float = 50.863565
    precipitation_var: str = "precipitation"

    @classmethod
    def from_config(cls, workdir, config_source, verbose=False, seed=None):
        config_data, _ = load_config(config_source)
        _ = workdir
        _ = verbose
        _ = seed
        return cls(precipitation_var=str(config_data.get("precipitation_var", "precipitation")))

    def kalman_step(self, ukf, measurements, meteo, pressure_at_bottom, site_id=None) -> float:
        _ = ukf
        _ = pressure_at_bottom
        _ = site_id
        scaling_factor = self._scaling_factor(measurements)
        precipitation = self._precipitation_window(meteo)
        return scaling_factor * self._mean_precipitation_m_per_day(precipitation)

    @staticmethod
    def _mean_precipitation_m_per_day(precipitation) -> float:
        """Mean precipitation rate of the window [m/day]; 0 for an empty window.

        The rate unit is read from the variable attribute `unit`, which zarr_fuse writes from the
        schema (e.g. APCP: "millimeter / second", chmi_stations_schema.yaml), and converted by pint.
        """
        if precipitation.size == 0:
            return 0.0
        assert "unit" in precipitation.attrs, (
            f"Precipitation variable {precipitation.name!r} has no 'unit' attribute (declared unit of the schema)"
        )
        rate = ureg.Quantity(float(precipitation.mean()), precipitation.attrs["unit"])
        return float(rate.to("meter / day").magnitude)

    @staticmethod
    def _scaling_factor(measurements) -> float:
        if measurements.sizes.get("date_time", 0) == 0:
            return 0.1
        mean_moisture = float(measurements["moisture"].mean())
        saturation = np.clip(mean_moisture, 0.0, 1.0)
        return 0.1 + 0.7 * saturation

    def _precipitation_window(self, meteo):
        precipitation = meteo[self.precipitation_var]
        if precipitation.sizes.get("date_time", 0) == 0:
            return precipitation

        end_time = np.datetime64(precipitation["date_time"].values[-1], "m")
        start_time = end_time - np.timedelta64(48, "h")
        return precipitation.sel(date_time=slice(start_time, end_time))

    def set_kalman_filter(self, kalman_R_matrix):
        return kalman_R_matrix

    def save_results(self):
        return None
        
        
@attrs.define(frozen=True)
class ConstantWeatherSite:
    site_id: int
    longitude: float
    latitude: float
    velocity: float


@attrs.define
class Model1DConstantWeather:
    composed: 'ComposedData'
    site: ConstantWeatherSite

    @classmethod
    def load_inputs(cls, composed, site_ids: list[int], config: dict) -> dict[int, ConstantWeatherSite]:
        """Inputs of the 1D sites: site_id -> ConstantWeatherSite (from the config, no store)."""
        _ = composed
        site_mapping = cls._site_mapping(config)
        return {
            int(site_id): ConstantWeatherSite(
                site_id=int(site_id),
                longitude=float(site_mapping[int(site_id)]["longitude"]),
                latitude=float(site_mapping[int(site_id)]["latitude"]),
                velocity=float(site_mapping[int(site_id)]["velocity"]),
            )
            for site_id in site_ids
        }

    @classmethod
    def from_config(cls, composed, site_id: int, config: dict, inputs: ConstantWeatherSite) -> "Model1DConstantWeather":
        """Model of one site; `inputs` from `load_inputs`."""
        _ = config
        assert inputs.site_id == site_id, f"Inputs of site {inputs.site_id} given to site {site_id}"
        return cls(composed=composed, site=inputs)

    @staticmethod
    def _site_mapping(config: dict) -> dict[int, dict]:
        raw_site_ids = config["site_ids"]
        assert isinstance(raw_site_ids, list), "model_1d.site_ids must be a list"
        raw_sites = config["sites"]
        assert isinstance(raw_sites, list), "model_1d.sites must be a list"
        assert len(raw_site_ids) == len(raw_sites), (
            "model_1d.sites length must match model_1d.site_ids length for Model1DConstantWeather"
        )
        site_mapping = {}
        for raw_site_id, site_cfg in zip(raw_site_ids, raw_sites):
            site_id = int(raw_site_id)
            assert isinstance(site_cfg, dict), "Each model_1d.sites item must be a mapping"
            for required_key in ("longitude", "latitude", "velocity"):
                assert required_key in site_cfg, (
                    f"Missing '{required_key}' in model_1d.sites entry for site_id={site_id}"
                )
            site_mapping[site_id] = site_cfg
        return site_mapping

    @property
    def longitude(self):
        return self.site.longitude

    @property
    def latitude(self):
        return self.site.latitude

    def step(self, start_time, target_time, pressure_at_bottom):
        _ = start_time
        _ = target_time
        _ = pressure_at_bottom
        return self.site.velocity

    def save_results(self):
        return None


SurfaceKalman = KalmanFilter
KalmanMock = SurfaceMock
KalmanScalingMock = SurfaceScalingMock


@attrs.define
class Model1D:
    composed: 'ComposedData'
    site_id: int
    moisture_sigma: float
    data: Model1DData
    kalman: KalmanFilter | SurfaceMock | SurfaceScalingMock

    @classmethod
    def load_inputs(cls, composed, site_ids: list[int], config: dict) -> dict[int, Model1DData]:
        """Inputs of the 1D sites: site_id -> Model1DData, read from the ZARR stores.

        Runs in the main process, see `Model1DData.load_sites`.
        """
        return Model1DData.load_sites(site_ids, composed, config)

    @classmethod
    def from_config(cls, composed, site_id: int, config: dict, inputs: Model1DData) -> "Model1D":
        """Model of one site; `inputs` from `load_inputs` (no store is opened here)."""
        kalman_class = resolve_named_class(
            config['kalman_class_name'],
            (KalmanFilter, SurfaceKalman, SurfaceMock, SurfaceScalingMock, KalmanMock, KalmanScalingMock),
        )
        data = inputs
        if kalman_class is KalmanFilter:
            meas_config = Model1D.create_kalman_measurements_config(data, config)
            moisture_sigma = meas_config["kalman_config"]["train_measurements"]['moisture']["noise_level"]
        else:
            moisture_sigma = config["moisture_sigma"]
        # TODO: refactor Kalman into Multiple nested classes so Model1D will be just one possible call of
        # an inner Kalman implementation, make syntehtic case and reading measurements from file as different
        # measuerement source classes.
        # That would allow to 1. construct the Model1D measurement class and pass it to Kalman with the remaining config.

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
                moisture_sigma=float(moisture_sigma),
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
        LOG.debug(
            "[1D %s] Kalman input window: %s -> %s, pressure_at_bottom=%s",
            self.site_id,
            start_time,
            target_time,
            pressure_at_bottom,
        )
        measurements = dataset_time_slice(self.data.profiles_dataset, start_time, target_time)
        meteo = dataset_time_slice(self.data.surface_dataset, start_time, target_time)
        LOG.debug(
            "[1D %s] measurement sizes=%s meteo sizes=%s",
            self.site_id,
            measurements.sizes,
            meteo.sizes,
        )

        darcy_velocity = None
        if len(measurements) > 0:
            darcy_velocity = self.kalman.kalman_step(
                self.ukf,
                measurements,
                meteo,
                pressure_at_bottom,
                site_id=self.site_id,
            )
        # TODO: more detailed output and either send through Queue to 3D worker and
        # save to ZARR from there, or excersize zarr parallel write (preallocation and suitable chunking necessary)
        return darcy_velocity

    def save_results(self):
        self.kalman.save_results()
    @staticmethod
    def create_kalman_measurements_config(data, config):
        sensor_depth = np.squeeze(data.profiles_dataset["sensor_depth"].values)

        # Configure Kalman filter measurement settings for training
        # "moisture" is treated as the observed variable
        config["kalman_config"]["train_measurements"] = {
            "moisture": {
                # Convert sensor depth to centimeters and invert sign
                "z_pos": sensor_depth * -100,
                # Measurement noise level (e.g., standard deviation)
                "noise_level": config["kalman_config"]["measurements_noise_level"],
                # Type of noise distribution (e.g., Gaussian, uniform)
                "noise_distr_type": config["kalman_config"]["measurements_noise_distr_type"]
            }
        }
        return config
