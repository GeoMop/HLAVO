from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class AbstractModel(ABC):
    def __init__(self, static_config, workdir=None):
        """
        Initialize the model with configuration and workdir.
        :param config: dict with static parameters expected by the model.
        :param workdir:
        """

        pass

    @abstractmethod
    def run(self, init_pressure: np.ndarray,  precipitation_value: float, model_params: dict, stop_time: float):
        """
        Run the model with initial pressure profile up to stop_time.
        :param init_pressure: np.ndarray, must agree with get_node_z
        :param precipitation_value: TODO: move under model_params
        :param model_params: TODO: should be dict of fixed values not distribution spec.
        :param stop_time:
        :return: ??
        """
        pass

    @abstractmethod
    def get_data(self, time: float, data_name: str ="pressure") -> np.ndarray:
        """
        Get "pressure" or "saturation" vector for given time. Must agree with get_nodes_z.
        :param time:
        :param data_name:
        :return:
        """
        pass

    @abstractmethod
    def get_times(self) -> list:
        pass

    @abstractmethod
    def get_nodes_z(self) -> float:
        """
        Return the Z coordinates of the data points.
        Must agree with nodes.
        :return:
        """
        pass

    @abstractmethod
    def plot_pressure(self):
        # Deprecated. Use kalman_result plotting or RichardsSolverOutput plotting.
        pass
