"""
Couple of miscellaneous functions used in multiple places
"""

import numpy as np


def vec_len(v: np.ndarray) -> int | float:
    """
    Length of a vector
    :param v:
    :return:
    """
    return np.sqrt(np.sum(np.power(v, 2)))


def sq_len(v: np.ndarray) -> int | float:
    """
    The "squared" length of a vector
    :param v:
    :return:
    """
    return float(np.sum(np.power(v, 2)))


def clamp(a: int | float, b: int | float, num: int | float) -> int | float:
    """
    Clamps the given number (num) between a and b
    :param a:
    :param b:
    :param num:
    :return:
    """
    if num < a:
        return a
    if num > b:
        return b
    return num


def closely_equal(val1: int | float, val2: int | float,
                  epsilon: int | float = 1e-6) -> bool:
    """
    Checks if the two values are closely equal, where the "closeness"
    is defined by the value of epsilon.
    :param val1:
    :param val2:
    :param epsilon:
    :return:
    """
    return abs(val1 - val2) < epsilon
