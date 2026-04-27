from pathlib import Path

# import polars as pl
# import zarr_fuse

from hlavo.misc.aux_zarr_fuse import remove_storage
from hlavo.ingress.moist_profile.extract.profile_process import main as extract
from hlavo.ingress.moist_profile.extract.profile_process import read_storage

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "moist_profile"

if __name__ == '__main__':
    root_path = SCRIPT_DIR.parents[2]
    print(root_path)
    print(DATA_DIR)

    # Test data
    # extract(source_dir=root_path / "tests/ingress/moist_profile/20260201T205548_dataflow_grab",
    #      storage_path=root_path / "tests/ingress/moist_profile/test_storage")
    read_storage(storage_path=root_path / "tests/ingress/moist_profile/test_storage")

    # Debug test - empty data
    # extract(source_dir=DATA_DIR / "test_empty",
    #         storage_path=Path("storage_empty"))

    # Local zarr_fuse
    # DVC data sources for year 2025
    # 2025 01-06
    extract(source_dir=DATA_DIR/"20260301T224908_dataflow_grab",
            storage_path=Path("storage_2025"))
    # 2025 07-12
    # extract(source_dir=DATA_DIR/"20260301T225923_dataflow_grab",
    #         storage_path=Path("storage_2025"))
    read_storage(storage_path=Path("storage_2025"))


    # Remote S3 storage
    storage_url = "s3://hlavo-testing/profiles.zarr"
    # storage_url = "s3://hlavo-release/profiles.zarr"
    # remove_storage(schema_path=Path("../profile_schema.yaml"), storage_path=storage_url)
    # data from 2025 01-06
    # main(source_dir="../20260301T224908_dataflow_grab",
    #      storage_path=storage_url)
    # data from 2025 07-12
    # main(source_dir="../20260301T225923_dataflow_grab",
    #      storage_path=storage_url)
    read_storage(storage_url)
