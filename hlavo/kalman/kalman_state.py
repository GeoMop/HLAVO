from typing import Any, Dict, Tuple, Callable
import attrs
from scipy import special, stats, linalg
import numpy as np
from scipy.sparse import csr_matrix
from hlavo.kalman.auxiliary_functions import add_noise

"""
Representation of Kalman state and associated variables.

Includes:
- Definitions for scalar and field variables (GVar, GField)
- Transformation utilities for encoding/decoding non-Gaussian variables
- Structures for state and measurement encoding/decoding
- Support for correlated fields and calibration coefficients

Used to build the UKF state vector and measurement vector representations.
"""

#############################
# Interpolation Utilities
#############################

def build_linear_interpolator_matrix(node_z, obs_z):
    """
    Build a sparse matrix performing 1D linear interpolation from node values to observation points.

    Rows follow the order of `obs_z`, with constant extrapolation beyond boundaries.

    :param node_z: Sorted 1D array of node coordinates
    :param obs_z: 1D array of observation coordinates
    :return: Sparse CSR matrix M such that M @ node_values interpolates to obs_z
    """
    node_z = np.asarray(node_z)
    obs_z = np.asarray(obs_z)

    N = len(node_z)
    M_size = len(obs_z)

    data, row_idx, col_idx = [], [], []

    for j, obs in enumerate(obs_z):
        if obs <= node_z[0]:  # Left extrapolation
            data.append(1.0); row_idx.append(j); col_idx.append(0)
        elif obs >= node_z[-1]:  # Right extrapolation
            data.append(1.0); row_idx.append(j); col_idx.append(N - 1)
        else:
            i = np.searchsorted(node_z, obs) - 1
            alpha = (obs - node_z[i]) / (node_z[i + 1] - node_z[i])
            data.extend([1 - alpha, alpha])
            row_idx.extend([j, j])
            col_idx.extend([i, i + 1])

    return csr_matrix((data, (row_idx, col_idx)), shape=(M_size, N))

#############################
# Transform Functions
#############################

def log_minus_one_to_gauss(value):
    """
    Transform from physical to Gaussian space for variables defined as log(x - 1).

    :param value: Input value(s)
    :return: Transformed values
    """
    return np.log(value - 1)

def log_minus_one_from_gauss(value):
    """
    Inverse transform from Gaussian to physical space for log(x - 1) mapping.

    :param value: Gaussian-space value(s)
    :return: Transformed back to physical space
    """
    return 1 + np.exp(value)

def saturation_to_moisture(saturation, state=None):
    """
    Convert saturation to volumetric moisture given a state dictionary.

    :param saturation: Saturation values
    :param state: Optional state dictionary containing 'vG_Th_s' and 'vG_Th_r'
    :return: Moisture values
    """
    if state is None:
        return saturation
    theta_sat = state["vG_Th_s"]
    theta_res = state["vG_Th_r"]
    return theta_sat * (saturation - theta_res) + theta_res

def moisture_to_saturation(moisture, state=None):
    """
    Convert volumetric moisture to saturation given a state dictionary.

    :param moisture: Moisture values
    :param state: Optional state dictionary containing 'vG_Th_s' and 'vG_Th_r'
    :return: Saturation values
    """
    if state is None:
        return moisture
    theta_sat = state["vG_Th_s"]
    theta_res = state["vG_Th_r"]
    return (moisture + theta_res) / theta_sat + theta_res

#############################
# Transform Wrappers
#############################

OneWayTransform = Callable[[np.ndarray], np.ndarray]
TwoWayTransform = Tuple[OneWayTransform, OneWayTransform]

class TwoArgWrapper:
    """Wrapper to allow transforms to optionally accept a 'state' argument."""
    def __init__(self, func):
        self.func = func
    def __call__(self, value, state=None):
        return self.func(value)

# Common transformations usable in Kalman variable definitions
transforms = dict(
    lognormal=(TwoArgWrapper(np.log), TwoArgWrapper(np.exp)),
    saturation_to_moisture=(TwoArgWrapper(moisture_to_saturation), TwoArgWrapper(saturation_to_moisture)),
    identity=(TwoArgWrapper(np.array), TwoArgWrapper(np.array)),
    log_minus_one=(TwoArgWrapper(log_minus_one_to_gauss), TwoArgWrapper(log_minus_one_from_gauss)),
    logit=(TwoArgWrapper(special.logit), TwoArgWrapper(special.expit))
)

#############################
# Gaussian Variables
#############################

@attrs.define
class GVar:
    """
    Scalar Gaussian variable.

    Represents a normally distributed quantity with mean, std, and process noise.
    """
    mean: float
    std: float
    Q: float
    ref: None
    z_pos: None
    transform: TwoWayTransform = transforms['identity']

    @classmethod
    def from_dict(cls, n_nodes, data: Dict[str, Any]) -> 'GVar':
        """
        Create a GVar from configuration dictionary.

        :param n_nodes: Number of nodes (unused for scalar)
        :param data: Configuration dictionary with mean/std or confidence interval
        :return: GVar instance
        """
        if 'mean_std' in data:
            mean, std = data['mean_std']
        elif 'conf_int' in data:
            conf_int = data['conf_int']
            if len(conf_int) == 2:
                conf_int.append(0.95)
            l, u, p = conf_int
            mean = (l + u) / 2.0
            z = stats.norm.ppf(1.0 - (1.0 - p) / 2.0)
            std = (u - l) / (2.0 * z)

        transform = transforms[data.get('transform', 'identity')]
        data_Q = (data['rel_std_Q'] * mean) ** 2
        z_pos = data.get("z_pos", None)
        return cls(mean=mean, std=std, Q=data_Q, ref=data['ref'], z_pos=z_pos, transform=transform)

    def size(self):
        """
        Return number of scalar elements (always 1).

        :return: 1
        """
        return 1

    @property
    def transform_to_gauss(self):
        return self.transform[0]

    @property
    def transform_from_gauss(self):
        return self.transform[1]

    @property
    def Q_full(self):
        """
        Return full covariance matrix (1x1).

        :return: Diagonal numpy array [[Q]]
        """
        return np.diag([self.Q])

    def init_state(self, nodes_z):
        """
        Return mean and covariance initialization for Kalman filter.

        :param nodes_z: Spatial nodes (unused)
        :return: Tuple (mean_array, cov_matrix)
        """
        return np.array([self.mean]), np.array([[self.std ** 2]])

    def encode(self, value):
        """
        Encode a scalar value to Gaussian space.

        :param value: Scalar value
        :return: Encoded numpy array
        """
        return self.transform_to_gauss(np.array([value]))

    def decode(self, value):
        """
        Decode a Gaussian-space value to physical space.

        :param value: Encoded numpy array
        :return: Decoded scalar
        """
        return self.transform_from_gauss(value)[0]

#############################
# Gaussian Field
#############################

@attrs.define
class FieldMeanLinear:
    """Linear mean profile defined by top and bottom values."""
    top: float
    bottom: float

    def make(self, node_z):
        """
        Construct linear mean field over depth.

        :param node_z: Array of node z-coordinates
        :return: Numpy array of mean values at each node
        """
        return np.linspace(self.top, self.bottom, len(node_z))

@attrs.define
class FieldCovExponential:
    """Exponential covariance structure for spatially correlated field."""
    std: float
    corr_length: float = 0.0
    exp: float = 2.0

    def make(self, node_z):
        """
        Construct covariance matrix using exponential kernel.

        :param node_z: Array of node z-coordinates
        :return: Covariance matrix
        """
        z_range = np.max(node_z) - np.min(node_z)
        if self.corr_length < 1.0e-10 * z_range:
            return np.diag(self.std ** 2 * np.ones(len(node_z)))
        s = np.abs(node_z[:, None] - node_z[:, None]) / self.corr_length
        return self.std ** 2 * np.exp(-s ** self.exp)

@attrs.define
class GField:
    """
    Gaussian correlated field (1D).

    Represents a random field with linear mean and exponential covariance.
    """
    mean: FieldMeanLinear
    cov: FieldCovExponential
    ref: None
    Q: float
    _size: int

    @classmethod
    def from_dict(cls, size, data: Dict[str, Any]) -> 'GField':
        """
        Create a correlated Gaussian field from configuration.

        :param size: Field size (number of nodes)
        :param data: Field configuration dictionary
        :return: GField instance
        """
        if 'mean_linear' in data:
            mean = FieldMeanLinear(**data['mean_linear'])
        if 'cov_exponential' in data:
            cov = FieldCovExponential(**data['cov_exponential'])

        mean_value = abs(data['mean_linear']['top'])
        data_Q = (data['rel_std_Q'] * mean_value) ** 2
        return cls(mean, cov, data.get('ref', None), data_Q, size)

    def size(self):
        return self._size

    @property
    def Q_full(self):
        return np.diag(self.size() * [self.Q])

    def init_state(self, el_centers_z):
        """
        Initialize mean and covariance field for Kalman state.

        :param el_centers_z: Element center z-coordinates
        :return: Tuple (mean_array, cov_matrix)
        """
        return self.mean.make(el_centers_z), self.cov.make(el_centers_z)

    def encode(self, value):
        return value

    def decode(self, value):
        return value

#############################
# Measurement Variable
#############################

@attrs.define
class Measure:
    """
    Represents a single measurement type (e.g., pressure, saturation, velocity).
    Handles noise and transforms.
    """
    z_pos: np.ndarray
    noise_level: float
    noise_distr_type: str
    interp: np.ndarray
    transform: TwoWayTransform = transforms['identity']

    @classmethod
    def from_dict(cls, nodes_z, data: Dict[str, Any]) -> 'Measure':
        """
        Create a measurement from configuration dictionary.

        :param nodes_z: Model nodes (used to compute interpolation matrix)
        :param data: Measurement configuration dictionary
        :return: Measure instance
        """
        data["interp"] = build_linear_interpolator_matrix(nodes_z, data["z_pos"])
        data["transform"] = transforms[data.get('transform', 'identity')]
        return cls(**data)

    def size(self):
        return len(self.z_pos)

    @property
    def ref(self):
        return np.zeros(self.size())

    @property
    def Q_full(self):
        return np.diag(self.size() * [1e-12])

    def init_state(self, nodes_z):
        return np.zeros(self.size()), 1e-12 * np.eye(self.size())

    def encode(self, value, state=None, noisy=False):
        """
        Encode measurement values (optionally add noise).

        :param value: Measurement values
        :param state: Optional state for nonlinear transforms
        :param noisy: Whether to add random noise
        :return: Encoded measurement array
        """
        if self.transform:
            value = self.transform[0](value, state)
        if noisy:
            value = add_noise(value, noise_level=self.noise_level, distr_type=self.noise_distr_type)
        return value

    def decode(self, value, state=None):
        """
        Decode measurement values back to physical domain.

        :param value: Encoded measurement array
        :param state: Optional state for inverse transform
        :return: Decoded measurement array
        """
        if self.transform:
            value = self.transform[1](value, state)
        return value

#############################
# State and Measurement Structures
#############################

class StateStructure(dict):
    """
    Container representing the full Kalman filter state.

    Encodes and decodes dictionaries of variables into 1D state vectors.
    """

    @staticmethod
    def _resolve_var_class(key):
        if key.endswith('_field'):
            return GField
        elif key.endswith('_meas'):
            return Measure
        elif key.endswith('calibration_coeffs'):
            return 'CalibrationCoeffs'
        else:
            return GVar

    def __init__(self, n_nodes, var_cfg):
        super().__init__({
            key: self._resolve_var_class(key).from_dict(n_nodes, val)
            for key, val in var_cfg.items()
        })

    def size(self):
        """
        Total size of the encoded state vector.

        :return: Integer length
        """
        return sum(var.size() for var in self.values())

    def compose_Q(self):
        """
        Assemble block-diagonal process noise covariance matrix Q.

        :return: 2D numpy array
        """
        Q_blocks = (var.Q_full for var in self.values())
        return linalg.block_diag(*Q_blocks)

    def compose_ref_dict(self):
        """
        Compose dictionary of reference values for each variable.

        :return: Dict of reference arrays
        """
        return {key: var.ref for key, var in self.items()}

    def compose_init_state(self, el_centers_z):
        """
        Compose concatenated initial mean and covariance from all state variables.

        :param el_centers_z: Element center coordinates
        :return: Tuple (mean_array, cov_matrix)
        """
        mean_list, cov_list = zip(*(var.init_state(el_centers_z) for var in self.values()))
        mean = np.concatenate(mean_list)
        cov = linalg.block_diag(*cov_list)
        return mean, cov

    def encode_state(self, value_dict):
        """
        Encode a dictionary of variable values into a single 1D state vector.

        :param value_dict: Dict of variable_name -> values
        :return: 1D numpy array
        """
        components = [var.encode(value_dict[key]) for key, var in self.items()]
        return np.concatenate(components)

    def decode_state(self, state_vector):
        """
        Decode a 1D state vector into a dictionary of variable values.

        :param state_vector: Flattened state array
        :return: Dict of variable_name -> decoded values
        """
        offset = np.cumsum([var.size() for var in self.values()])
        return {
            key: var.decode(state_vector[var_off - var.size(): var_off])
            for var_off, (key, var) in zip(offset, self.items())
        }

    def get_calibration_coeffs_z_positions(self):
        """
        Get z-positions of calibration coefficients if present.

        :return: List of z-positions or empty list
        """
        calibration_coeffs = self.get("calibration_coeffs", [])
        if not calibration_coeffs:
            return []
        return [c.z_pos for c in calibration_coeffs.GVar_coeffs]

class MeasurementsStructure(dict):
    """
    Container for all measurement variables.

    Handles encoding/decoding and optional application of calibration coefficients.
    """

    def __init__(self, nodes_z, var_cfg):
        el_z = (nodes_z[1:] + nodes_z[:-1]) / 2.0
        measure_dict = {}
        for key, val in var_cfg.items():
            if key == "velocity":
                measure_dict[key] = Measure.from_dict(nodes_z, val)
            else:
                measure_dict[key] = Measure.from_dict(el_z, val)
        super().__init__(measure_dict)

    def size(self):
        return sum(var.size() for var in self.values())

    def z_positions(self, meas_key):
        return [var.z_pos for key, var in self.items() if meas_key == key]

    def encode(self, value_dict, state=None, noisy=False):
        """
        Encode dictionary of measurement values into 1D vector.

        :param value_dict: Dict of measurement_name -> values
        :param state: Optional state for nonlinear transforms
        :param noisy: Whether to add noise
        :return: 1D numpy array
        """
        components = [var.encode(value_dict[key], state, noisy) for key, var in self.items()]
        return np.concatenate(components)

    def mult_calibration_coef(self, measurements_struct, measurements, calibration_coefs, calibration_coeffs_z_positions):
        """
        Apply calibration coefficients to measurements in-place.

        :param measurements_struct: MeasurementsStructure defining z-positions
        :param measurements: Dict of measurement_name -> arrays
        :param calibration_coefs: Calibration coefficient values
        :param calibration_coeffs_z_positions: Positions corresponding to coefficients
        :return: Updated measurements dict
        """
        for key, values in measurements.items():
            for idx, z_position in enumerate(measurements_struct.z_positions(meas_key=key)[0]):
                callibration_mult_coeff = calibration_coefs[
                    np.squeeze(np.where(calibration_coeffs_z_positions == z_position)[0])
                ]
                measurements[key][idx] *= callibration_mult_coeff
        return measurements

    def decode(self, meas_vector):
        """
        Decode measurement vector into dictionary of named measurements.

        :param meas_vector: Flattened measurement array
        :return: Dict of measurement_name -> decoded arrays
        """
        offset = np.cumsum([var.size() for var in self.values()])
        return {
            key: var.decode(meas_vector[var_off - var.size(): var_off])
            for var_off, (key, var) in zip(offset, self.items())
        }

@attrs.define
class CalibrationCoeffs:
    """
    Represents a collection of calibration coefficients, one per measurement depth.
    """
    ref: None
    GVar_coeffs: None

    @classmethod
    def from_dict(cls, n_nodes, data):
        """
        Build CalibrationCoeffs object from list of GVar configurations.

        :param n_nodes: Number of mesh nodes
        :param data: List of dicts defining each calibration coefficient
        :return: CalibrationCoeffs instance
        """
        GVar_coeffs = []
        ref = []
        for position_coeff in data:
            ref.append(position_coeff["ref"])
            GVar_coeffs.append(GVar.from_dict(n_nodes=n_nodes, data=position_coeff))
        return cls(ref=ref, GVar_coeffs=GVar_coeffs)

    def size(self):
        return sum(var.size() for var in self.GVar_coeffs)

    @property
    def Q_full(self):
        return np.diag([var.Q for var in self.GVar_coeffs])

    @property
    def mean(self):
        return np.array([var.mean for var in self.GVar_coeffs])

    @property
    def std(self):
        return np.array([var.std for var in self.GVar_coeffs])

    def init_state(self, nodes_z):
        """
        Initialize mean and covariance for calibration coefficients.

        :param nodes_z: Node coordinates (unused)
        :return: Tuple (mean_array, cov_matrix)
        """
        return (
            np.array([var.mean for var in self.GVar_coeffs]),
            np.diag(np.array([[var.std ** 2 for var in self.GVar_coeffs]]).flatten())
        )

    def encode(self, values):
        """
        Encode list of coefficient values into Gaussian space.

        :param values: List or array of coefficient values
        :return: Encoded 1D numpy array
        """
        components = [var.encode(values[idx]) for idx, var in enumerate(self.GVar_coeffs)]
        return np.concatenate(components)

    def decode(self, values):
        """
        Decode Gaussian-space values back to coefficient values.

        :param values: Encoded 1D numpy array
        :return: List of decoded coefficient values
        """
        return [var.decode([values[idx]]) for idx, var in enumerate(self.GVar_coeffs)]

    @property
    def transform_from_gauss(self):
        """Return combined inverse transform for all coefficients."""
        def transform_from_gauss_coeffs(values):
            values = np.squeeze(values)
            return np.squeeze([
                var.transform_from_gauss([values[idx]]) for idx, var in enumerate(self.GVar_coeffs)
            ])
        return transform_from_gauss_coeffs

    @property
    def transform_to_gauss(self):
        """Return combined forward transform for all coefficients."""
        def transform_to_gauss_coeffs(values):
            return np.squeeze([
                var.transform_to_gauss([values[idx]]) for idx, var in enumerate(self.GVar_coeffs)
            ])
        return transform_to_gauss_coeffs
