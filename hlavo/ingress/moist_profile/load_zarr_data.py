import zarr_fuse as zf
from pathlib import Path


def load_measurments_data(scheme_file):
    """
    Load measurement profile dataset from a Zarr scheme file.
    :param str | Path scheme_file: Path to the measurement scheme file.
    :return: Dataset containing measurement profiles.
    """
    scheme_file_path = Path(scheme_file)
    root = zf.open_store(scheme_file_path)

    profiles = root["Uhelna"]["profiles"]
    return profiles.dataset


def load_meteo_data(scheme_file):
    """
    Load meteorological dataset from a Zarr scheme file.
    :param str | Path scheme_file: Path to the meteorological scheme file.
    :return: Dataset containing meteorological data.
    """
    scheme_file_path = Path(scheme_file)
    root = zf.open_store(scheme_file_path)


    meteo_data = root["Uhelna"]["parflow"]["version_01"] #root["chmi_aladin_10m"]
    #meteo_data = root["parflow_input"]
    return meteo_data.dataset
