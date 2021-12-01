import numpy as np

from .dual import Dual

__all__ = ["sin", "cos", "tan", "exp", "log", "sqrt", "arcsin", "arccos", "arctan", "logistic"]


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

def arcsin(x):
    if x.val**2 >= 1:
        raise ValueError(f"Derivative of arcsin(x) is undefined for x = {x.val}")

    return Dual(np.arcsin(x.val),x.der/(np.sqrt(1-x.val**2)))

def arccos(x):
    if x.val**2 >= 1:
        raise ValueError(f"Derivative of arccos(x) is undefined for x = {x.val}")

    return Dual(np.arccos(x.val),-x.der/(np.sqrt(1-x.val**2)))

def arctan(x):
    return Dual(np.arctan(x),x.der/(1+x.val**2))

def logistic(x):
    g = lambda z: 1/(1+np.exp(-z))
    return Dual(g(x.val),x.der*g(x.val)*(1-g(x.val)))