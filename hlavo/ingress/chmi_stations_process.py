import sys
import argparse
from hlavo.misc.aux_zarr_fuse import remove_storage
from hlavo.ingress.meteo_playground.chmi_stations.config import (
    CHMI_STATIONS_SCHEMA_PATH
)

from hlavo.ingress.meteo_playground.chmi_stations.data_processing import main as data_process


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process scraped data and update zarr_fuse storage.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Remove the storage at first.",
    )
    return parser


if __name__ == '__main__':
    start_date = "2020-01-01T00:00:00"
    end_date = "2025-12-31T23:59:59"

    args = _build_parser().parse_args(sys.argv[1:])
    print(f"Remove storage: {args.force}")

    # possibly remove old storage
    if args.force:
        remove_storage(schema_path=CHMI_STATIONS_SCHEMA_PATH)

    data_process(start_date, end_date)
