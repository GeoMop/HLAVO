from __future__ import annotations

import sys
from pathlib import Path

import yaml

from zarr_fuse.units import ureg


def collect_units(obj, *, path=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}/{key}" if path else str(key)
            if key == "unit":
                yield next_path, value
            else:
                yield from collect_units(value, path=next_path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from collect_units(value, path=f"{path}[{index}]")


def main(argv: list[str]) -> int:
    """
    Takes schema path as argument and
    - finds units in schema and tries to parse them with the lib
    - explicitly checks unit "C" which is Coulomb, not Celsius, possibly warns
    - prints the unit and its found name
    - raises error if unit is unknown

    If no argument given, it prints out all available units from the lib.
    :param argv:
    :return: 0 if passes ok, 1 otherwise
    """
    if not argv:
        print(f"Available zarr_fuse units: {len(ureg._units)}")
        for unit_name in sorted(ureg._units):
            print(unit_name)
        print("\nUsage: list_zarr_fuse_units.py <schema.yaml>")
        return 0

    schema_path = Path(argv[0])
    with schema_path.open(encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    print(f"\nChecking units in {schema_path}")
    problems_found = False
    for unit_path, unit_value in collect_units(schema):
        if unit_value == "C":
            print(f"WARNING: {unit_path} uses 'C', which pint interprets as coulomb, not Celsius.")
            problems_found = True
            continue
        try:
            parsed = ureg.parse_units(unit_value)
            print(f"OK: {unit_path} -> {unit_value} -> {parsed}")
        except Exception as exc:
            print(f"ERROR: {unit_path} -> {unit_value}: {type(exc).__name__}: {exc}")
            problems_found = True

    if not problems_found:
        print("No invalid or suspicious units found.")
    return 1 if problems_found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
