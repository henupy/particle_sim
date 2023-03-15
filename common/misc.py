"""
Couple of miscellaneous functions used in multiple places
"""

import numpy as np

# For typing
numeric = int | float


def vec_len(v: np.ndarray) -> numeric:
    """
    Length of a vector
    :param v:
    :return:
    """
    return np.sqrt(np.sum(np.power(v, 2)))


def sq_len(v: np.ndarray) -> numeric:
    """
    The 'squared' length of a vector
    :param v:
    :return:
    """
    return float(np.sum(np.power(v, 2)))


def clamp(a: numeric, b: numeric, num: numeric) -> numeric:
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
