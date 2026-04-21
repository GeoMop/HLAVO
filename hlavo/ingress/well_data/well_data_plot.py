#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   David Flanderka

from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # for interactive graphs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml

matplotlib.rcParams['hatch.linewidth'] = 6


_PLOT_CONFIG = {
    "cum_draw": {
        "dataset_name": "water_draw",
        "x_column": "date",
        "title": "Water draw well: '{well_id}'",
    },
    "water_level": {
        "dataset_name": "water_levels",
        "x_column": "date_time",
        "title": "Water levels well: '{well_id}'",
    },
}


def _create_work_dir():
    """
    Create workdir if doesn't exist, return path.
    """
    script_dir = Path(__file__).parent
    workdir = script_dir / "workdir"
    workdir.mkdir(exist_ok=True)
    return workdir


def _schema_units():
    """
    Read plotting units from wells_schema.yaml.
    """
    schema_path = Path(__file__).parent / "wells_schema.yaml"
    with schema_path.open("r", encoding="utf-8") as file_in:
        schema = yaml.safe_load(file_in)

    uhelna_schema = schema["Uhelna"]
    return {
        "cum_draw": uhelna_schema["water_draw"]["VARS"]["cum_draw"]["unit"],
        "water_level": uhelna_schema["water_levels"]["VARS"]["water_level"]["unit"],
    }


def _to_pandas(df):
    """
    Convert a dataframe-like object returned by zarr-fuse to pandas.
    """
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df


def _plot_ax_variable(df, well_id, variable_name):
    """
    Plot one zarr-fuse variable for one well into a single matplotlib figure.
    """
    assert variable_name in _PLOT_CONFIG

    df_pandas = _to_pandas(df)
    df_filtered = df_pandas.loc[df_pandas["well_id"] == well_id]

    plot_config = _PLOT_CONFIG[variable_name]
    units = _schema_units()

    figure, axis = plt.subplots(figsize=(10, 5))
    if not df_filtered.empty:
        df_filtered.plot(
            x=plot_config["x_column"],
            y=variable_name,
            ax=axis,
        )

    axis.set_title(plot_config["title"].format(well_id=well_id))
    axis.set_xlabel("Date")
    axis.set_ylabel(f"{variable_name} [{units[variable_name]}]")
    axis.grid(True)
    figure.tight_layout()
    return figure


def pdf_plot_all(pdf_file, df_draw, df_water_levels, water_level_well_ids):
    """
    Write a single PDF containing draw plot followed by water level plots.
    """
    workdir = _create_work_dir()
    full_path = workdir / pdf_file

    with PdfPages(full_path) as pdf:
        figure = _plot_ax_variable(df_draw, "Uh-draw", "cum_draw")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        for well_id in water_level_well_ids:
            figure = _plot_ax_variable(df_water_levels, well_id, "water_level")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
