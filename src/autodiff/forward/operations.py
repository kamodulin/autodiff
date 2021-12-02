import numpy as np

from .dual import Dual

__all__ = [
    "sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
    "exp", "log", "sqrt", "logistic"
]


def sin(x):
    try:
        val = np.sin(x.val)
        der = np.cos(x.val) * x.der
        return Dual(val, der)
    except AttributeError:
        return np.sin(x)


def cos(x):
    try:
        val = np.cos(x.val)
        der = -np.sin(x.val) * x.der
        return Dual(val, der)
    except AttributeError:
        return np.cos(x)


def tan(x):
    try:
        if np.isclose(np.cos(x.val), 0):
            raise ValueError(
                f"Derivative of tan(x) is undefined for x = {x.val}")
        val = np.tan(x.val)
        der = x.der / (np.cos(x.val)**2)
        return Dual(val, der)
    except AttributeError:
        return np.tan(x)


def sinh(x):
    try:
        val = np.sinh(x.val)
        der = np.cosh(x.val) * x.der
        return Dual(val, der)
    except AttributeError:
        return np.sinh(x)


def cosh(x):
    try:
        val = np.cosh(x.val)
        der = np.sinh(x.val) * x.der
        return Dual(val, der)
    except AttributeError:
        return np.cosh(x)


def tanh(x):
    try:
        val = np.tanh(x.val)
        der = (1 - np.tanh(x.val)**2) * x.der
        return Dual(val, der)
    except AttributeError:
        return np.tanh(x)


def arcsin(x):
    try:
        if abs(x.val) > 1:
            raise ValueError(
                f"Derivative of arcsin(x) is undefined for x = {x.val}")
        val = np.arcsin(x.val)
        der = x.der / (np.sqrt(1 - x.val**2))
        return Dual(val, der)
    except AttributeError:
        return np.arcsin(x)


def arccos(x):
    try:
        if abs(x.val) > 1:
            raise ValueError(
                f"Derivative of arccos(x) is undefined for x = {x.val}")
        val = np.arccos(x.val)
        der = -x.der / (np.sqrt(1 - x.val**2))
        return Dual(val, der)
    except AttributeError:
        return np.arccos(x)


def arctan(x):
    try:
        val = np.arctan(x.val)
        der = x.der / (1 + x.val**2)
        return Dual(val, der)
    except AttributeError:
        return np.arctan(x)


def exp(x):
    try:
        val = np.exp(x.val)
        der = np.exp(x.val) * x.der
        return Dual(val, der)
    except AttributeError:
        return np.exp(x)


def log(x):
    try:
        if x.val <= 0:
            raise ValueError(f"Log of x is undefined for x = {x.val}")
        val = np.log(x.val)
        der = x.der / x.val
        return Dual(val, der)
    except AttributeError:
        return np.log(x)


def sqrt(x):
    try:
        if x.val < 0:
            raise ValueError(f"Derivative of sqrt(x) is undefined for x < 0")
        val = np.sqrt(x.val)
        der = (0.5 / val) * x.der
        return Dual(val, der)
    except AttributeError:
        return np.sqrt(x)


def logistic(x):
    g = lambda z: 1 / (1 + np.exp(-z))
    try:
        val = g(x.val)
        der = x.der * g(x.val) * (1 - g(x.val))
        return Dual(val, der)
    except AttributeError:
        return g(x)
