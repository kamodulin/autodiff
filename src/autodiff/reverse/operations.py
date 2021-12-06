import numpy as np

from .node import Node

__all__ = [
    "sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh",
    "arcsin", "arccos", "arctan", "logistic"
]


def sin(x):
    """
    Return the sine of x.

    Parameters
    ----------
    x : int, float, Node

    Returns
    -------
    y : float or Node

    Examples
    --------
    >>> ad.sin(np.pi / 2)
    1.0

    >>> x = ad.Node(np.pi / 2)
    >>> ad.sin(x)
    Node(1.0)
    """
    try:
        child = Node(np.sin(x.val))
        x._addChildren(np.cos(x.val), child)
        return child
    except AttributeError:
        return np.sin(x)


def cos(x):
    """
    Return the cosine of x.

    Parameters
    ----------
    x : int, float, Node

    Returns
    -------
    y : float or Node

    Examples
    --------
    >>> ad.cos(np.pi / 2)
    0.0

    >>> x = ad.Node(np.pi / 2)
    >>> ad.cos(x)
    Node(0)
    """
    try:
        child = Node(np.cos(x.val))
        x._addChildren(-np.sin(x.val), child)
        return child
    except AttributeError:
        return np.cos(x)


def tan(x):
    """
    Return the tangent of x.

    Parameters
    ----------
    x : int, float, Node

    Returns
    -------
    y : float or Node

    Notes
    -----
    The derivative of tan(x) is undefined when the cosine of x is zero.

    Examples
    --------
    >>> ad.tan(np.pi / 4)
    1.0

    >>> x = ad.Node(np.pi / 4)
    >>> ad.tan(x)
    Node(1.0)

    Derivative undefined:
    
    >>> x = ad.Node(np.pi / 2)
    >>> ad.tan(x)
    Traceback (most recent call last):
    ...
    ValueError: Derivative of tan(x) is undefined for x = 1.5707963267948966
    """
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
    """
    Return the hyperbolic sine of x.

    Parameters
    ----------
    x : int, float, Node

    Returns
    -------
    y : float or Node

    Examples
    --------
    >>> ad.sinh(1)
    1.1752011936438014

    >>> x = ad.Node(2)
    >>> ad.sinh(x)
    Node(3.626860407847019)
    """
    try:
        child = Node(np.sinh(x.val))
        x._addChildren(np.cosh(x.val), child)
        return child
    except AttributeError:
        return np.sinh(x)


def cosh(x):
    """
    Return the hyperbolic cosine of x.

    Parameters
    ----------
    x : int, float, Node

    Returns
    -------
    y : float or Node

    Examples
    --------
    >>> ad.cosh(1)
    1.5430806348152437

    >>> x = ad.Node(2)
    >>> ad.cosh(x)
    Node(3.7621956910836314)
    """
    try:
        child = Node(np.cosh(x.val))
        x._addChildren(np.sinh(x.val), child)
        return child
    except AttributeError:
        return np.cosh(x)


def tanh(x):
    """
    Return the hyperbolic tangent of x.

    Parameters
    ----------
    x : int, float, Node

    Returns
    -------
    y : float or Node

    Examples
    --------
    >>> ad.tanh(1)
    0.7615941559557649

    >>> x = ad.Node(1)
    >>> ad.tanh(x)
    Node(0.7615941559557649)
    """
    try:
        child = Node(np.tanh(x.val))
        x._addChildren((1 - np.tanh(x.val)**2), child)
        return child
    except AttributeError:
        return np.tanh(x)


def arcsin(x):
    """
    Return the inverse sine of x.

    Parameters
    ----------
    x : int, float, Node

    Returns
    -------
    y : float or Node

    Notes
    -----
    The derivative of arcsin(x) is undefined when x is not in the range (-1, 1).

    Examples
    --------
    >>> ad.arcsin(1)
    1.5707963267948966

    >>> x = ad.Node(0.5)
    >>> ad.arcsin(x)
    Node(0.5235987755982989)

    Derivative undefined:

    >>> x = ad.Node(1)
    >>> ad.arcsin(x)
    Traceback (most recent call last):
    ...
    ValueError: Derivative of arcsin(x) is undefined for x = 1
    """
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
    """
    Return the inverse cosine of x.

    Parameters
    ----------
    x : int, float, Node

    Returns
    -------
    y : float or Node

    Notes
    -----
    The derivative of arccos(x) is undefined when x is not in the range (-1, 1).

    Examples
    --------
    >>> ad.arccos(1)
    0.0
    
    >>> x = ad.Node(0.5)
    >>> ad.arccos(x)
    Node(1.0471975511965979)

    Derivative undefined:

    >>> x = ad.Node(1)
    >>> ad.arccos(x)
    Traceback (most recent call last):
    ...
    ValueError: Derivative of arcsin(x) is undefined for x = 1
    """
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
    """
    Return the inverse tangent of x.

    Parameters
    ----------
    x : int, float, Node

    Returns
    -------
    y : float or Node

    Examples
    --------
    >>> ad.arctan(1)
    0.7853981633974483

    >>> x = ad.Node(1)
    >>> ad.arctan(x)
    Node(0.7853981633974483)
    """
    try:
        child = Node(np.arctan(x.val))
        x._addChildren((1 / (1 + x.val**2)), child)
        return child
    except AttributeError:
        return np.arctan(x)


def exp(x):
    """
    Return the exponential of x.

    Parameters
    ----------
    x : int, float, Node

    Returns
    -------
    y : float or Node

    Examples
    --------
    >>> ad.exp(1)
    2.718281828459045

    >>> x = ad.Node(1)
    >>> ad.exp(x)
    Node(2.718281828459045)
    """
    try:
        child = Node(np.exp(x.val))
        x._addChildren(np.exp(x.val), child)
        return child
    except AttributeError:
        return np.exp(x)


def log(x, base=np.e):
    """
    Return the logarithm of x of any base, default to natural log.

    Parameters
    ----------
    x : int, float, Dual
    base : int, float

    Returns
    -------
    out : float or Dual

    Notes
    -----
    The log of x is undefined when x is less than or equal to 0.

    Examples
    --------
    >>> ad.log(2)
    0.6931471805599453

    >>> x = ad.Node(2)
    >>> ad.log(x)
    Node(0.6931471805599453)

    >>> x = ad.Node(2)
    >>> ad.log(x, base=10)
    Node(0.30102999566398114)

    >>> x = ad.Node(2)
    >>> ad.log(x, base=2)
    Node(1.0)

    Derivative undefined:

    >>> x = ad.Node(0)
    >>> ad.log(x)
    Traceback (most recent call last):
    ...
    ValueError: Log of x is undefined for x = 0

    Error when base is non-positive:

    >>> x = ad.Node(2)
    >>> ad.log(x, base=0)
    Traceback (most recent call last):
    ...
    ValueError: Cannot have non-positive base
    """
    if base <= 0:
        raise ValueError(f'Cannot have non-positive base')
    try:
        if x.val <= 0:
            raise ValueError(f"Log of x is undefined for x = {x.val}")
        child = Node((np.log(x.val) / np.log(base)))
        x._addChildren((1 / (x.val * np.log(base))), child)
        return child
    except AttributeError:
        if x <= 0:
            raise ValueError(f"Log of x is undefined for x = {x}")
        return np.log(x) / np.log(base)


def sqrt(x):
    """
    Return the square root of x.

    Parameters
    ----------
    x : int, float, Node

    Returns
    -------
    y : float or Node

    Notes
    -----
    The square root of x is undefined when x is less than 0.

    Examples
    --------
    >>> ad.sqrt(4)
    2.0

    >>> x = ad.Node(4)
    >>> ad.sqrt(x)
    Node(2.0)

    Derivative undefined:

    >>> x = ad.Node(-1)
    >>> ad.sqrt(x)
    Traceback (most recent call last):
    ...
    ValueError: Derivative of sqrt(x) is undefined for x < 0
    """
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
    """
    Return the logistic function of x.

    Parameters
    ----------
    x : int, float, Node

    Returns
    -------
    y : float or Node

    Examples
    --------
    >>> ad.logistic(1)
    0.7310585786300049

    >>> x = ad.Node(3)
    >>> ad.logistic(x)
    Node(0.9525741268224334)
    """
    g = lambda z: 1 / (1 + np.exp(-z))
    try:
        child = Node(g(x.val))
        x._addChildren(g(x.val) * (1 - g(x.val)), child)
        return child
    except AttributeError:
        return g(x)