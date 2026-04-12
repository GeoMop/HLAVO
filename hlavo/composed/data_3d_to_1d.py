import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Data3DTo1D:
    date_time: np.datetime64
    # For which date time we provide the data.
    site_id: int
    # For which site id we provide the data. The site id is an integer that uniquely identifies a site.
    pressure_head: float
    # The pressure head [m] at the bottom of the 1D model from the last 3D model step.
