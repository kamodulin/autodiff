import numpy as np

from .dual import Dual

__all__ = ["sin", "cos", "tan", "exp", "log", "sqrt"]


def sin(x):
    return Dual(np.sin(x.val), np.cos(x.val) * x.der)


def cos(x):
    return Dual(np.cos(x.val), -np.sin(x.val) * x.der)


def tan(x):
    try:
        return Dual(np.tan(x.val), x.der / (np.cos(x.val)**2))
    except ZeroDivisionError:
        raise ValueError(f"Derivative of tan(x) is undefined for x = {x.val}")


def exp(x):
    ...


def log(x):
    ...


def sqrt(x):
    ...
