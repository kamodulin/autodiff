import numpy as np

from .node import Node

__all__ = ["sin", "cos", "tan", "exp", "log", "sqrt","sinh","cosh","tanh","arcsin","arccos","arctan","logistic"]


def sin(x):
    if isinstance(x, (int, float)):
        return np.sin(x)

    child = Node(np.sin(x.val))
    x._addChildren(np.cos(x.val),child)
    return child

def cos(x):
    if isinstance(x, (int, float)):
        return np.cos(x)

    child = Node(np.cos(x.val))
    x._addChildren(-np.sin(x.val),child)
    return child

def tan(x):
    if isinstance(x, (int, float)):
        return np.tan(x)

    if np.isclose(np.cos(x.val), 0):
        raise ValueError(f"Derivative of tan(x) is undefined for x = {x.val}")

    child = Node(np.tan(x.val))
    x._addChildren((1/(np.cos(x.val)**2)),child)
    return child

def exp(x):
    if isinstance(x, (int, float)):
        return np.exp(x)

    child = Node(np.exp(x.val))
    x._addChildren(np.exp(x.val),child)
    return child


def log(x):
    if isinstance(x, (int, float)):
        if x <= 0:
            raise ValueError(f"Log of x is undefined for x = {x}")
        return np.log(x)

    if x.val <= 0:
        raise ValueError(f"Log of x is undefined for x = {x.val}")
    
    child = Node(np.log(x.val))
    x._addChildren((1/x.val),child)
    return child

def sqrt(x):
    if isinstance(x, (int, float)):
        if x < 0:
            raise ValueError(f"Derivative of sqrt(x) is undefined for x < 0")
        return np.sqrt(x)

    if x.val < 0:
        raise ValueError(f"Derivative of sqrt(x) is undefined for x < 0")

    child = Node(np.sqrt(x.val))
    x._addChildren(0.5/np.sqrt(x.val),child)
    return child


def logistic(x):
    try:
        g = lambda z: 1/(1+np.exp(-z))
        child = Node(g(x.val))
        x._addChildren(g(x.val) * (1 - g(x.val)), child)
        return child
    except AttributeError:
        return 1/(1+np.exp(-x))


def sinh(x):
    try:
        child = Node(np.sinh(x.val))
        x._addChildren(np.cosh(x.val),child)
        return child
    except AttributeError:
        return np.sinh(x)

def cosh(x):
    try:
        child = Node(np.cosh(x.val))
        x._addChildren(np.sinh(x.val),child)
        return child
    except AttributeError:
        return np.cosh(x)

def tanh(x):
    try:
        child = Node(np.tanh(x.val))
        x._addChildren((1 - np.tanh(x.val)**2),child)
        return child
    except AttributeError:
        return np.tanh(x)
        
        
# Inverse trig functions:
def arcsin(x):
    try:
        child = Node(np.arcsin(x.val))
        x._addChildren((1 / np.sqrt(1 - x.val ** 2)),child)
        return child
    except AttributeError:
        return np.arcsin(x)
        
def arccos(x):
    try:
        child = Node(np.arccos(x.val))
        x._addChildren((-1 / np.sqrt(1 - x.val ** 2)),child)
        return child
    except AttributeError:
        return np.arcsin(x)
        
def arctan(x):
    try:
        child = Node(np.arctan(x.val))
        x._addChildren((1 / (1 + x.val ** 2)),child)
        return child
    except AttributeError:
        return np.arcsin(x)
        