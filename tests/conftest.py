"""
Common test configuration for all test subdirectories.
Put here only those things that can not be done through command line options and pytest.ini file.
"""

import pytest
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# add tests dir to sys path in order to get access to the 'fixtures' module.
this_source_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(this_source_dir)

#https://stackoverflow.com/questions/37563396/deleting-py-test-tmpdir-directory-after-successful-test-case
# @pytest.fixture(scope='session')
# def temporary_dir(tmpdir_factory):
#     img = compute_expensive_image()
#     fn = tmpdir_factory.mktemp('data').join('img.png')
#     img.save(str(fn))
#     return fn


@pytest.fixture
def local_mock_profile_surface_store(tmp_path):
    profiles_store = tmp_path / "profiles.zarr"
    surface_store = tmp_path / "surface.zarr"

    date_time = np.array(
        ["2025-03-06T00:00:00", "2025-03-06T01:00:00", "2025-03-06T02:00:00"],
        dtype="datetime64[ns]",
    )
    site_id = np.array([1], dtype=np.int32)
    depth_level = np.array([0], dtype=np.int32)

    profiles = xr.Dataset(
        data_vars={
            "moisture": (
                ("date_time", "site_id", "depth_level"),
                np.full((date_time.size, site_id.size, depth_level.size), 0.2),
            ),
            "sensor_depth": (("depth_level",), np.array([0.1], dtype=float)),
            "longitude": (
                ("date_time", "site_id"),
                np.tile(np.array([[14.88]], dtype=float), (date_time.size, 1)),
            ),
            "latitude": (
                ("date_time", "site_id"),
                np.tile(np.array([[50.86]], dtype=float), (date_time.size, 1)),
            ),
        },
        coords={
            "date_time": date_time,
            "site_id": site_id,
            "depth_level": depth_level,
        },
    )
    profiles_path = profiles_store / "Uhelna" / "profiles"
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_zarr(profiles_path)

    surface = xr.Dataset(
        data_vars={
            "precipitation": (("date_time", "site_id"), np.zeros((date_time.size, site_id.size))),
            "temperature": (("date_time", "site_id"), np.full((date_time.size, site_id.size), 273.15)),
        },
        coords={
            "date_time": date_time,
            "site_id": site_id,
        },
    )
    surface_path = surface_store / "Uhelna" / "parflow" / "version_01"
    surface_path.parent.mkdir(parents=True, exist_ok=True)
    surface.to_zarr(surface_path)

    profiles_schema = tmp_path / "profile_schema.yaml"
    profiles_schema.write_text(
        "\n".join(
            [
                "ATTRS:",
                f'  STORE_URL: "file://{profiles_store}"',
                "Uhelna:",
                "  profiles: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    surface_schema = tmp_path / "surface_schema.yaml"
    surface_schema.write_text(
        "\n".join(
            [
                "ATTRS:",
                f'  STORE_URL: "file://{surface_store}"',
                "Uhelna:",
                "  parflow:",
                "    version_01: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "profiles": profiles_schema,
        "surface": surface_schema,
    }
