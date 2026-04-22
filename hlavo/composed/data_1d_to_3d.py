import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Data1DTo3D:
    date_time: np.datetime64['m']
    # For which date time we provide the data.
    site_id: int
    # For which site id we provide the data. The site id is an integer that uniquely identifies a site.
    longitude: float
    latitude: float
    # The longitude and latitude of the site.
    velocity: float
    # The Darcy velocity at the bottom of the 1D model.

