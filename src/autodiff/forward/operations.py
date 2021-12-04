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
    return Dual(np.exp(x.val), np.exp(x.val) * x.der)


def log(x):
    if x.val <= 0:
        raise ValueError(f"Log of x is undefined for x = {x.val}")
    return Dual(np.log(x.val), x.der / x.val)


def sqrt(x):
    if x.val < 0:
        raise ValueError(f"Derivative of sqrt(x) is undefined for x < 0")
    return Dual(np.sqrt(x.val), 0.5 / np.sqrt(x.val) * x.der)


# Inverse trig functions:
def arcsin(x):
    der = 1 / np.sqrt(1 - x.val ** 2) * x.der
    val = np.arcsin(x.val)
    return Dual(val, der)
        
def arccos(x):
    der = -1 / np.sqrt(1 - x.val ** 2) * x.der
    val = np.arccos(x.val)
    return Dual(val, der)
        
def arctan(x):
    der = 1 / (1 + x.val ** 2) * x.der
    val = np.arctan(x.val)
    return Dual(val, der)
        
