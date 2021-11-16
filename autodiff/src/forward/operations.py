import numpy as np

from .dual import Dual

__all__ = ["sin", "cos", "tan", "exp", "log", "sqrt"]


def sin(x):
    return Dual(np.sin(x.val), np.cos(x.val) * x.der)


def cos(x):
    return Dual(np.cos(x.val), -np.sin(x.val) * x.der)


def tan(x):
    if np.isclose(np.cos(x.val), 0):
        raise ValueError(f"Derivative of tan(x) is undefined for x = {x.val}")
    return Dual(np.tan(x.val), x.der / (np.cos(x.val)**2))

def exp(x):
    ...


def log(x):
    if x.val <= 0:
        raise ValueError(f"Log of x is undefined for x = {x.val}")
    return Dual(np.log(x.val), x.der / x.val)


def sqrt(x):
    ...
