import numpy as np

from .node import Node

__all__ = [
    "sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh",
    "arcsin", "arccos", "arctan", "logistic"
]


def sin(x):
    try:
        child = Node(np.sin(x.val))
        x._addChildren(np.cos(x.val), child)
        return child
    except AttributeError:
        return np.sin(x)


def cos(x):
    try:
        child = Node(np.cos(x.val))
        x._addChildren(-np.sin(x.val), child)
        return child
    except AttributeError:
        return np.cos(x)


def tan(x):
    try:
        if np.isclose(np.cos(x.val), 0):
            raise ValueError(
                f"Derivative of tan(x) is undefined for x = {x.val}")
        child = Node(np.tan(x.val))
        x._addChildren((1 / (np.cos(x.val)**2)), child)
        return child
    except AttributeError:
        return np.tan(x)


def sinh(x):
    try:
        child = Node(np.sinh(x.val))
        x._addChildren(np.cosh(x.val), child)
        return child
    except AttributeError:
        return np.sinh(x)


def cosh(x):
    try:
        child = Node(np.cosh(x.val))
        x._addChildren(np.sinh(x.val), child)
        return child
    except AttributeError:
        return np.cosh(x)


def tanh(x):
    try:
        child = Node(np.tanh(x.val))
        x._addChildren((1 - np.tanh(x.val)**2), child)
        return child
    except AttributeError:
        return np.tanh(x)


def arcsin(x):
    try:
        if abs(x.val) >= 1:
            raise ValueError(
                f"Derivative of arcsin(x) is undefined for x = {x.val}")
        child = Node(np.arcsin(x.val))
        x._addChildren((1 / np.sqrt(1 - x.val**2)), child)
        return child
    except AttributeError:
        return np.arcsin(x)


def arccos(x):
    try:
        if abs(x.val) >= 1:
            raise ValueError(
                f"Derivative of arcsin(x) is undefined for x = {x.val}")
        child = Node(np.arccos(x.val))
        x._addChildren((-1 / np.sqrt(1 - x.val**2)), child)
        return child
    except AttributeError:
        return np.arccos(x)


def arctan(x):
    try:
        child = Node(np.arctan(x.val))
        x._addChildren((1 / (1 + x.val**2)), child)
        return child
    except AttributeError:
        return np.arctan(x)


def exp(x):
    try:
        child = Node(np.exp(x.val))
        x._addChildren(np.exp(x.val), child)
        return child
    except AttributeError:
        return np.exp(x)


def log(x):
    try:
        if x.val <= 0:
            raise ValueError(f"Log of x is undefined for x = {x.val}")
        child = Node(np.log(x.val))
        x._addChildren((1 / x.val), child)
        return child
    except AttributeError:
        if x <= 0:
            raise ValueError(f"Log of x is undefined for x = {x}")
        return np.log(x)


def sqrt(x):
    try:
        if x.val < 0:
            raise ValueError(f"Derivative of sqrt(x) is undefined for x < 0")
        child = Node(np.sqrt(x.val))
        x._addChildren(0.5 / np.sqrt(x.val), child)
        return child
    except AttributeError:
        if x < 0:
            raise ValueError(f"Derivative of sqrt(x) is undefined for x < 0")
        return np.sqrt(x)


def logistic(x):
    g = lambda z: 1 / (1 + np.exp(-z))
    try:
        child = Node(g(x.val))
        x._addChildren(g(x.val) * (1 - g(x.val)), child)
        return child
    except AttributeError:
        return g(x)