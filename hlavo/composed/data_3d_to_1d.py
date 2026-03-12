import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Data3DTo1D:
    date_time: np.datetime64
    site_id: int
    pressure_head: Any
