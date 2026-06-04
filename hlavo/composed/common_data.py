import attrs
from pathlib import Path
import numpy as np

@attrs.define
class ComposedData:
    seed: int
    start: np.datetime64['s']
    end: np.datetime64['s']
    workdir: Path
    config_dir: Path

    @classmethod
    def from_config(cls, workdir, config: dict, config_path):
        return  ComposedData(
            seed = config.get("seed", np.random.randint(0, 1_000_000)),
            start = np.datetime64(config["start_datetime"], 's'),
            end = np.datetime64(config["end_datetime"], 's'),
            workdir = workdir,
            config_dir = config_path.parent,
    )
    def relative_resolve(self, relative_path: str | Path) -> Path:
        return (self.config_dir / relative_path).resolve()