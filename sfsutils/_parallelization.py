"""
Parallelization and bounds-checking utilities.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import logging
import math
from typing import Callable, Dict, Literal, Sequence, Tuple

import multiprocess as mp
import numpy as np
from tqdm import tqdm

from .settings import Settings

# get logger
logger = logging.getLogger('sfsutils').getChild('Parallelization')


def parallelize(
        func: Callable,
        data: Sequence,
        parallelize: bool = True,
        pbar: bool = None,
        desc: str = None,
        dtype: type = object,
        wrap_array: bool = True
) -> np.ndarray:
    """
    Parallelize given function or execute sequentially.

    :param parallelize: Whether to parallelize
    :param data: Data to iterate over
    :param func: Function to apply to each element of data
    :param pbar: Whether to show a progress bar
    :param desc: Description for progress bar
    :param dtype: Data type of the returned array
    :param wrap_array: Whether to wrap the result in a numpy array
    :return: List of results
    """
    n = len(data)

    if parallelize and n > 1 and Settings.parallelize is not False:
        # parallelize
        iterator = mp.Pool().imap(func, data)
    else:
        # sequentialize
        iterator = map(func, data)

    # whether to show a progress bar
    if pbar is True or (pbar is None and n > 1):
        iterator = tqdm(iterator, total=n, disable=Settings.disable_pbar, desc=desc)

    if wrap_array:
        return np.array(list(iterator), dtype=dtype)

    return list(iterator)


def check_bounds(
        bounds: Dict[str, Tuple[float, float]],
        params: Dict[str, float],
        fixed_params: Dict[str, float] = {},
        percentile: float = 1,
        scale: Literal['lin', 'log'] = 'lin'
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, Tuple[float, float, float]]]:
    """
    Issue warnings if the passed parameters are close to the specified bounds.

    :param bounds: The bounds to check against.
    :param params: The parameters to check.
    :param fixed_params: The fixed parameters.
    :param percentile: The percentile threshold to consider a parameter close to the bounds.
    :param scale: Scale type: 'lin' for linear and 'log' for logarithmic.
    :return: Tuple of dictionaries of parameters close to the lower and upper bounds, i.e. (lower, value, upper).
    """
    near_lower = {}
    near_upper = {}

    def transform(value: float, to_scale: Literal['lin', 'log']) -> float:
        """
        Transform a value to the specified scale.

        :param value: The value to transform.
        :param to_scale: The scale to transform to.
        :return: The transformed value.
        """
        if to_scale == 'log':
            return math.log(value) if value > 0 else -float('inf')

        return value

    for key, value in params.items():
        # get base name
        name = key.split('.')[-1]

        # get bounds
        lower, upper = bounds[name]

        # transform values
        _lower = transform(lower, scale)
        _upper = transform(upper, scale)
        _value = transform(value, scale)

        if key not in fixed_params:
            # relative proximity is only defined when both bounds are set (finite range)
            if _lower is not None and _upper is not None:
                if (_value - _lower) / (_upper - _lower) <= percentile / 100:
                    near_lower[key] = (lower, value, upper)

                if (_upper - _value) / (_upper - _lower) <= percentile / 100:
                    near_upper[key] = (lower, value, upper)

    return near_lower, near_upper
