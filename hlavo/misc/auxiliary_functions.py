import numpy as np
import scipy as sc
from functools import reduce


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        print("isPD")
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        print("while not isPD")
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = sc.linalg.cholesky(B)
        return True
    except sc.linalg.LinAlgError:
        return False


def sqrt_func(x):
    print("sqrt func call")
    try:
        result = sc.linalg.cholesky(x)
    except:# np.linalg.LinAlgError:
        #x = (x + x.T)/2
        x = nearestPD(x)
        e_val, e_vec = np.linalg.eigh(x)
        print("e_val ", e_val)
        result = sc.linalg.cholesky(x)
        print("result ", result)
    return result


def get_nested_attr(obj, attr):
    """
    Access nested attributes of an object using a dot-separated string.
    """
    return reduce(getattr, attr.split('.'), obj)


def set_nested_attr(obj, attr, value):
    """
    Set the value of a nested attribute of an object using a dot-separated string.
    """
    pre, _, post = attr.rpartition('.')
    return setattr(get_nested_attr(obj, pre) if pre else obj, post, value)

def set_nested_attrs(obj, dict):
    """
    Use all keys of given dict to set nested attributes.
    """
    for key, val in dict.items():
        set_nested_attr(obj, key, val)


def add_noise(data_array, noise_level=0.1, distr_type="uniform", seed=12345):
    if len(data_array) > 0:

        if distr_type == "uniform":
            # print("uniform ")
            # print("noise level ", noise_level)
            # print("input data array ", data_array)

            lower_bound = -np.abs(noise_level * data_array)
            upper_bound = np.abs(noise_level * data_array)

            #print("orig lower bound: {}, upper bound: {}".format(lower_bound, upper_bound))

            # if std is not None:
            #     noise_level = std
            #     # std = np.array(std)
            #     # range = std * np.sqrt(12)
            #     # lower_bound = - range/2
            #     # upper_bound = range / 2
            # lower_bound = -np.abs(noise_level * data_array)
            # upper_bound = np.abs(noise_level * data_array)

            #print("lower bound: {}, upper bound: {}".format(lower_bound, upper_bound))

            noise = np.random.uniform(lower_bound, upper_bound)
            data_array = data_array + noise

        elif distr_type == "gaussian":
            #orig_value_sign = np.sign(data_array)

            # print("noise level ", noise_level)
            # print("type noise level ", type(noise_level))
            # print("data array ", data_array)

            #if std is None:
            #    std = np.abs(data_array * noise_level)

            value_noise = np.random.normal(0, noise_level)
            print("value: {}, noise: {}, value + noise: {}".format(data_array, value_noise, data_array + value_noise))

            data_array = data_array + value_noise

            # different_signs_indices = np.where(np.sign(data_array) != orig_value_sign)[0]
            # for idx in different_signs_indices:
            #     data_array[idx] *= -1

    print("data array ", data_array)

    return data_array
