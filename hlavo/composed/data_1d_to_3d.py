import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Data1DTo3D:
    date_time: np.datetime64['m']
    site_id: int
    longitude: float
    latitude: float
    velocity: Any # Units not yet known

