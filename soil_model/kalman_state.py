from typing import Any, Dict, Tuple, Union, List, Callable
import attrs
from scipy import special
from scipy import stats
import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from auxiliary_functions import add_noise
"""
Representation of Kalman state and associated variables:
- initial mean and covariance: x, P
- process noise covariance: Q (but its discutable to associate it with state as it describes noise in the model)
- reference value: ref; used to compute syntetic measurements
===
- decoding the state vector to state dictionary and encoding back to vector
- Support for lornormal transform.
- Basic support for correlated fields.

"""

def build_linear_interpolator_matrix(node_z, obs_z):
    """
    Build a sparse matrix M that performs 1D linear interpolation
    from the values on node_z to the points obs_z, with constant
    extrapolation beyond the boundaries.

    This version uses an explicit loop over obs_z to ensure that
    rows in the interpolation matrix follow the order of obs_z.

    Parameters
    ----------
    node_z : sorted 1D array of shape (N,)
        The x-coordinates of the nodes where values are given.
    obs_z  : 1D array of shape (M,)
        The x-coordinates of the observation points to interpolate.

    Returns
    -------
    M : (M, N) sparse CSR matrix
        So that M @ node_values (shape (N,)) produces
        interpolated values at obs_z (shape (M,)).
    """
    node_z = np.asarray(node_z)
    obs_z  = np.asarray(obs_z)

    N = len(node_z)
    M_size = len(obs_z)

    data = []
    row_idx = []
    col_idx = []

    for j, obs in enumerate(obs_z):
        # Out-of-bounds handling
        if obs <= node_z[0]:  # Left extrapolation (constant)
            data.append(1.0)
            row_idx.append(j)
            col_idx.append(0)
        elif obs >= node_z[-1]:  # Right extrapolation (constant)
            data.append(1.0)
            row_idx.append(j)
            col_idx.append(N - 1)
        else:
            # Find interval idx such that node_z[i] <= obs < node_z[i+1]
            i = np.searchsorted(node_z, obs) - 1

            # Compute interpolation weight (alpha)
            alpha = (obs - node_z[i]) / (node_z[i+1] - node_z[i])

            # Store two values in sparse matrix (linear interpolation)
            data.extend([1 - alpha, alpha])
            row_idx.extend([j, j])
            col_idx.extend([i, i + 1])

    # Create sparse matrix
    M = csr_matrix((data, (row_idx, col_idx)), shape=(M_size, N))

    return M


def log_minus_one_to_gauss(value):
    return np.log(value - 1)

def log_minus_one_from_gauss(value):
    return 1 + np.exp(value)

def saturation_to_moisture(saturation, state=None):
    if state is None:
        return saturation
    print("decoded_state ", state)
    theta_sat = state["vG_Th_s"]
    theta_res = state["vG_Th_r"]
    return theta_sat * (saturation - theta_res) + theta_res

def moisture_to_saturation(moisture, state=None):
    if state is None:
        return moisture
    print("state ", state)
    theta_sat = state["vG_Th_s"]
    theta_res = state["vG_Th_r"]
    return (moisture + theta_res)/theta_sat + theta_res

#############################
# Transforms
#############################
OneWayTransform = Callable[[np.ndarray], np.ndarray]
TwoWayTransform = Tuple[OneWayTransform, OneWayTransform]

# To allow 'saturation_to_moisture' and 'moisture_to_saturation' to be used as transformations,
# all other transforms must also accept a 'state' parameter.
class TwoArgWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, value, state=None):
        return self.func(value)


transforms = dict(
    lognormal = (
        TwoArgWrapper(np.log),
        TwoArgWrapper(np.exp)
    ),
    saturation_to_moisture = (
        TwoArgWrapper(moisture_to_saturation),
        TwoArgWrapper(saturation_to_moisture)
    ),
    identity = (
        TwoArgWrapper(np.array),
        TwoArgWrapper(np.array)
    ),
    log_minus_one = (
        TwoArgWrapper(log_minus_one_to_gauss),
        TwoArgWrapper(log_minus_one_from_gauss)
    ),
    logit = (
        TwoArgWrapper(special.logit),
        TwoArgWrapper(special.expit)
    )
)
# OneWayTransform = Callable[[np.ndarray], np.ndarray]
# TwoWayTransform = Tuple[OneWayTransform, OneWayTransform]
# transforms = dict(
#     lognormal = (np.log, np.exp),   # to gauss, from gauss
#     identity = (np.array, np.array),
#     log_minus_one = (log_minus_one_to_gauss, log_minus_one_from_gauss),
#     logit = (special.logit, special.expit)
# )
#############################
# Variable Classes
#############################

@attrs.define
class GVar:
    """
    Normaly distributed variable.
    TODO:
    Could possibly be unified with the field to support vector variables.
    Difference is just specific kind of correlation matrix.
    """
    mean: float
    std: float
    Q: float
    ref: None
    z_pos: None
    transform: TwoWayTransform = transforms['identity']

    @classmethod
    def from_dict(cls, n_nodes, data: Dict[str, Any]) -> 'KalmanVar':
        if 'mean_std' in data:
            mean, std = data['mean_std']
        elif 'conf_int' in data:
            conf_int: List[float] = data['conf_int']
            if len(conf_int) == 2:
                conf_int.append(0.95)   # Default confidence level (95%)
            l, u, p = conf_int

            # Compute the mean as the midpoint of the interval
            mean = (l + u) / 2.0
            # Compute the standard deviation from the confidence interval
            z = stats.norm.ppf(1.0 - (1.0 - p) / 2.0)  # z-score for the confidence level
            std = (u - l) / (2.0 * z)
        transform = transforms[data.get('transform', 'identity')]
        print("mean ", mean)
        data_Q = (data['rel_std_Q'] * mean) ** 2
        print("data Q ", data_Q)

        z_pos = data.get("z_pos", None)

        # Optional z_pos
        return cls(mean=mean, std=std, Q=data_Q, ref=data['ref'], z_pos=z_pos, transform=transform)

    def size(self) -> int:
        return 1

    @property
    def transform_to_gauss(self):
        return self.transform[0]

    @property
    def transform_from_gauss(self):
        return self.transform[1]

    @property
    def Q_full(self) -> np.ndarray:
        return np.diag([self.Q])

    def init_state(self, nodes_z):
        return np.array([self.mean]), np.array([[self.std ** 2]])

    def encode(self, value: float) -> np.ndarray:
        return self.transform_to_gauss(np.array([value]))

    def decode(self, value: np.ndarray) -> float:
        return self.transform_from_gauss(value)[0]

MeanField = Union['FieldMeanLinear']
@attrs.define
class FieldMeanLinear:
    top: float
    bottom: float

    def make(self, node_z):
        return np.linspace(self.top, self.bottom, len(node_z))

CovField = Union['FieldCovExponential']
@attrs.define
class FieldCovExponential:
    std: float
    corr_length: float = 0.0
    exp: float = 2.0

    def make(self, node_z):
        z_range = np.max(node_z) - np.min(node_z)
        if self.corr_length < 1.0e-10 * z_range:
            return np.diag(self.std ** 2 * np.ones(len(node_z)))
        s = np.abs(node_z[:, None]-node_z[:,None]) / self.corr_length
        cov = self.std ** 2 * np.exp(- s ** self.exp)
        return cov

@attrs.define
class GField:
    """
    Gaussian correlation field (1D).
    """
    mean: MeanField
    cov: CovField
    ref: None
    Q: float
    _size: int
    #transform: Callable[[float], float] = None

    @classmethod
    def from_dict(cls, size, data: Dict[str, Any]) -> 'GField':
        # JB TODO: check form wring config keyword make more structured config
        # to simplify implementation
        if 'mean_linear' in data:
            mean = FieldMeanLinear(**data['mean_linear'])
        if 'cov_exponential' in data:
            cov = FieldCovExponential(**data['cov_exponential'])

        #print("mean linear ", np.abs((data['mean_linear']['top'] + data['mean_linear']['bottom'])/2))
        mean_value = np.abs(data['mean_linear']['top'])
        data_Q = (data['rel_std_Q'] * mean_value)**2
        #print("data Q ", data_Q)

        return cls(mean, cov, data.get('ref', None), data_Q, size)

    def size(self) -> int:
        return self._size

    @property
    def Q_full(self) -> np.ndarray:
        return np.diag(self.size() * [self.Q])

    def init_state(self, el_centers_z):
        assert len(el_centers_z) == self.size()
        return self.mean.make(el_centers_z), self.cov.make(el_centers_z)

    def encode(self, value: np.ndarray) -> np.ndarray:
        return value

    def decode(self, value: np.ndarray) -> np.ndarray:
        return value

@attrs.define
class Measure:
    """
    Auxiliary representing measurement in the state.
    """
    z_pos: np.ndarray
    noise_level: float
    noise_distr_type: str
    interp: np.ndarray
    transform: TwoWayTransform = transforms['identity']

    @classmethod
    def from_dict(cls, nodes_z, data: Dict[str, Any]) -> 'GField':
        # JB TODO: check form wring config keyword make more structured config
        # to simplify implementation
        data["interp"] = build_linear_interpolator_matrix(nodes_z, data["z_pos"])

        data["transform"] = transforms[data.get('transform', 'identity')]
        return cls(**data)

    def size(self) -> int:
        return len(self.z_pos)

    @property
    def ref(self) -> np.ndarray:
        return np.zeros(self.size())

    @property
    def Q_full(self) -> np.ndarray:
        return np.diag(self.size() * [1e-12])

    def init_state(self, nodes_z):
        return np.zeros(self.size()), 1e-12 * np.eye(self.size())

    def encode(self, value: np.ndarray, state=None, noisy:bool=False) -> np.ndarray:
        if self.transform is not None:
            value = self.transform[0](value, state)
        if noisy:
            value = add_noise(value, noise_level=self.noise_level, distr_type=self.noise_distr_type)
        return value

    def decode(self, value: np.ndarray, state=None) -> np.ndarray:
        if self.transform is not None:
            value = self.transform[1](value, state)
        return value


#############################
# Dictionary for declaring state structure and properties.
#############################
# class StateStructure:
#     @staticmethod
#     def _resolve_var_class(key):
#         if key.endswith('_field'):
#             return GField
#         elif key.endswith('_meas'):
#             return Measure
#         else:
#             return GVar
#
#     def __init__(self, n_nodes, var_cfg: Dict[str, Dict[str, Any]]):
#         self.var_dict : Dict[str, Union[GVar, GField,Measure]] = {
#             key: self._resolve_var_class(key).from_dict(n_nodes, val)
#             for key, val in var_cfg.items()
#         }
#
#     def __getitem__(self, item):
#         return self.var_dict[item]
#
#     def size(self):
#         return sum(var.size for var in self.var_dict.values())
#
#     def compose_Q(self) -> np.ndarray:
#         Q_blocks = (var.Q_full for var in self.var_dict.values())
#         return linalg.block_diag(*Q_blocks)
#
#     def compose_ref_dict(self) -> np.ndarray:
#         ref_dict = {key: var.ref for key, var in  self.var_dict.items()}
#         return ref_dict
#
#     def compose_init_state(self, nodes_z) -> np.ndarray:
#         mean_list, cov_list = zip(*(var.init_state(nodes_z) for var in  self.var_dict.values()))
#         mean = np.concatenate(mean_list)
#         cov = linalg.block_diag(*cov_list)
#         return mean, cov
#
#     def encode_state(self, value_dict):
#         """
#         Encodes a dict of GVariable objects into a single 1D state vector.
#
#         :param var_dict: Dict of {variable_name: GVariable}
#         :return: A 1D numpy array representing the concatenation of all variables' fields.
#         """
#         # Decide on a consistent order. Here we use the insertion or .values() order.
#         # You could also sort by name for consistency if needed.
#         components = [var.encode(value_dict[key]) for key, var in self.var_dict.items()]
#         return np.concatenate(components)
#
#
#     def decode_state(self, state_vector):
#         """
#         Decodes a 1D state vector into the existing GVariable objects.
#
#         :param var_dict: Dict of {variable_name: GVariable}
#         :param state_vector: 1D numpy array containing all the data in the same order used by encode_state.
#         """
#         offset = np.cumsum([var.size for var in self.var_dict.values()])
#         state_dict = {
#             key: var.decode(state_vector[var_off-var.size: var_off])
#             for var_off, (key, var) in zip(offset, self.var_dict.items())
#         }
#         return state_dict
#


class StateStructure(dict):
    """
    Kalman filter works with the state vector that could actualy be composed from distinct quantities of
    different name, scale and distribution
    Encode and decode the state dictionary to and from measurement vector.
    """

    @staticmethod
    def _resolve_var_class(key):
        if key.endswith('_field'):
            return GField
        elif key.endswith('_meas'):
            return Measure
        elif key.endswith('calibration_coeffs'):
            return CalibrationCoeffs
        else:
            return GVar

    def __init__(self, n_nodes, var_cfg: Dict[str, Dict[str, Any]]):
        super().__init__({
            key: self._resolve_var_class(key).from_dict(n_nodes, val)
            for key, val in var_cfg.items()
        })

    def from_dict(cls, n_nodes, data: Dict[str, Any]) -> 'KalmanVar':
        if 'mean_std' in data:
            mean, std = data['mean_std']
        elif 'conf_int' in data:
            conf_int: List[float] = data['conf_int']
            if len(conf_int) == 2:
                conf_int.append(0.95)   # Default confidence level (95%)
            l, u, p = conf_int

            # Compute the mean as the midpoint of the interval
            mean = (l + u) / 2.0
            # Compute the standard deviation from the confidence interval
            z = stats.norm.ppf(1.0 - (1.0 - p) / 2.0)  # z-score for the confidence level
            std = (u - l) / (2.0 * z)
        transform = transforms[data.get('transform', 'identity')]
        print("mean ", mean)
        data_Q = (data['rel_std_Q'] * mean) ** 2
        print("data Q ", data_Q)

        # Optional z_pos
        z_pos = data.get('z_pos', None)
        return cls(mean=mean, std=std, Q=data_Q, ref=data['ref'], z_pos=z_pos, transform=transform)

    def size(self):
        return sum(var.size() for var in self.values())

    def compose_Q(self) -> np.ndarray:
        Q_blocks = (var.Q_full for var in self.values())
        return linalg.block_diag(*Q_blocks)

    def compose_ref_dict(self) -> Dict[str, Any]:
        return {key: var.ref for key, var in self.items()}

    def compose_init_state(self, el_centers_z) -> np.ndarray:
        mean_list, cov_list = zip(*(var.init_state(el_centers_z) for var in self.values()))
        mean = np.concatenate(mean_list)
        cov = linalg.block_diag(*cov_list)

        return mean, cov

    def encode_state(self, value_dict: Dict[str, Any]) -> np.ndarray:
        components = [var.encode(value_dict[key]) for key, var in self.items()]
        return np.concatenate(components)

    def decode_state(self, state_vector: np.ndarray) -> Dict[str, Any]:
        """
        Decodes a 1D state vector into the existing GVariable objects.
        :param var_dict: Dict of {variable_name: GVariable}
        :param state_vector: 1D numpy array containing all the data in the same order used by encode_state.
        """
        offset = np.cumsum([var.size() for var in self.values()])
        state_dict = {
            key: var.decode(state_vector[var_off - var.size(): var_off])
            for var_off, (key, var) in zip(offset, self.items())
        }
        return state_dict

    def get_calibration_coeffs_z_positions(self):
        calibration_coeffs = self.get("calibration_coeffs", [])
        if len(calibration_coeffs) == 0:
            return []
        return [calib_coeff.z_pos for calib_coeff in calibration_coeffs.GVar_coeffs]


class MeasurementsStructure(dict):
    """
    Kalman filter works with measurement vector that could actualy be composed from distinct quantities of
    different name, scale and distribution
    Encode and decode the measurement dictionary to and from measurement vector.
    """
    def __init__(self, nodes_z, var_cfg: Dict[str, Dict[str, Any]]):
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

    def encode(self, value_dict: Dict[str, Any], state=None, noisy=False) -> np.ndarray:
        components = [var.encode(value_dict[key], state, noisy) for key, var in self.items()]
        return np.concatenate(components)

    def mult_calibration_coef(self, measurements_struct, measurements: Dict[str, Any], calibration_coefs, calibration_coeffs_z_positions) -> np.ndarray:
        for key, values in measurements.items():
            print("measurements_struct.z_positions(meas_key=key) ", measurements_struct.z_positions(meas_key=key))
            for idx, z_position in enumerate(measurements_struct.z_positions(meas_key=key)[0]):
                print("z_position ", z_position)
                print("np.where(calibration_coeffs_z_positions == z_position) ", np.where(calibration_coeffs_z_positions == z_position)[0])
                callibration_mult_coeff = calibration_coefs[np.squeeze(np.where(calibration_coeffs_z_positions == z_position)[0])]
                measurements[key][idx] *= callibration_mult_coeff

        print("measurements ", measurements)
        return measurements



    def decode(self, meas_vector: np.ndarray) -> Dict[str, Any]:
        """
        Decodes a 1D state vector into the existing GVariable objects.
        :param meas_vector: 1D numpy array containing all the data in the same order used by encode.
        """
        offset = np.cumsum([var.size() for var in self.values()])
        state_dict = {
            key: var.decode(meas_vector[var_off - var.size(): var_off])
            for var_off, (key, var) in zip(offset, self.items())
        }
        return state_dict

@attrs.define
class CalibrationCoeffs:
    ref: None
    GVar_coeffs: None

    @classmethod
    def from_dict(cls, n_nodes, data: Dict[str, Any]):
        print("data", data)

        #z_positions = data["z_positions"]

        GVar_coeffs = []
        ref = []
        for position_coeff in data:
            ref.append(position_coeff["ref"])
            GVar_coeffs.append(GVar.from_dict(n_nodes=n_nodes, data=position_coeff))

        # print("z_positions ", z_positions)
        # print("GVar_coeffs ", GVar_coeffs)

        return cls(ref=ref, GVar_coeffs=GVar_coeffs)

    def size(self):
        return sum(var.size() for var in self.GVar_coeffs)

    @property
    def Q_full(self) -> np.ndarray:
        return np.diag([var.Q for var in self.GVar_coeffs])

    @property
    def mean(self) -> np.ndarray:
        return np.array([var.mean for var in self.GVar_coeffs])

    @property
    def std(self) -> np.ndarray:
        return np.array([var.std for var in self.GVar_coeffs])

    def init_state(self, nodes_z):
        return np.array([var.mean for var in self.GVar_coeffs]), np.diag(np.array([[var.std**2 for var in self.GVar_coeffs]]).flatten())

    def encode(self, values) -> np.ndarray:
        print("encode values ", values)
        print("self.GVar_coeffs ", self.GVar_coeffs)
        components = [var.encode(values[idx]) for idx, var in enumerate(self.GVar_coeffs)]
        print("components ", components)
        return np.concatenate(components)

    def decode(self, values: np.ndarray) -> np.ndarray:
        print("values ", values)
        return [var.decode([values[idx]]) for idx, var in enumerate(self.GVar_coeffs)]

    @property
    def transform_from_gauss(self):
        def transform_from_gauss_coeffs(values):
            print("values ", values)
            values = np.squeeze(values)
            print("from gauss: [var.transform_from_gauss([values[idx]]) for idx, var in enumerate(self.GVar_coeffs)] ", np.squeeze([var.transform_from_gauss([values[idx]]) for idx, var in enumerate(self.GVar_coeffs)]))
            return np.squeeze([var.transform_from_gauss([values[idx]]) for idx, var in enumerate(self.GVar_coeffs)])
        return transform_from_gauss_coeffs

    @property
    def transform_to_gauss(self):
        def transform_to_gauss_coeffs(values):
            print("values ", values)
            print("to gauss: ", np.squeeze([var.transform_to_gauss([values[idx]]) for idx, var in enumerate(self.GVar_coeffs)]))
            return np.squeeze([var.transform_to_gauss([values[idx]]) for idx, var in enumerate(self.GVar_coeffs)])

        return transform_to_gauss_coeffs