import numpy as np

from .dual import Dual

__all__ = [
    "sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
    "exp", "log", "sqrt", "logistic"
]


def sin(x):
    """
    Return the sine of x.

    Parameters
    ----------
    x : int, float, Dual

    Returns
    -------
    y : float or Dual

    Examples
    --------
    >>> ad.sin(np.pi / 2)
    1.0

    >>> x = ad.Dual(np.pi / 2)
    >>> ad.sin(x)
    Dual(1.0, array([0.0]))
    """
    try:
        val = np.sin(x.val)
        der = np.cos(x.val) * x.der
        return Dual(val, der)
    except AttributeError:
        return np.sin(x)


def cos(x):
    """
    Return the cosine of x.

    Parameters
    ----------
    x : int, float, Dual

    Returns
    -------
    y : float or Dual

    Examples
    --------
    >>> ad.cos(np.pi / 2)
    0.0

    >>> x = ad.Dual(np.pi / 2)
    >>> ad.cos(x)
    Dual(0.0, array([-1.0]))
    """
    try:
        val = np.cos(x.val)
        der = -np.sin(x.val) * x.der
        return Dual(val, der)
    except AttributeError:
        return np.cos(x)


def tan(x):
    """
    Return the tangent of x.

    Parameters
    ----------
    x : int, float, Dual

    Returns
    -------
    y : float or Dual

    Notes
    -----
    The derivative of tan(x) is undefined when the cosine of x is zero.

    Examples
    --------
    >>> ad.tan(np.pi / 4)
    1.0

    >>> x = ad.Dual(np.pi / 4)
    >>> ad.tan(x)
    Dual(1.0, array([2.0]))

    Derivative undefined:

    >>> x = ad.Dual(np.pi / 2)
    >>> ad.tan(x)
    Traceback (most recent call last):
    ...
    ValueError: Derivative of tan(x) is undefined for x = 1.5707963267948966
    """
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
    """
    Return the hyperbolic sine of x.

    Parameters
    ----------
    x : int, float, Dual

    Returns
    -------
    y : float or Dual

    Examples
    --------
    >>> ad.sinh(1)
    1.1752011936438014

    >>> x = ad.Dual(2, 1)
    >>> ad.sinh(x)
    Dual(3.6268604078470186, array([3.76219569]))
    """
    try:
        val = np.sinh(x.val)
        der = np.cosh(x.val) * x.der
        return Dual(val, der)
    except AttributeError:
        return np.sinh(x)


def cosh(x):
    """
    Return the hyperbolic cosine of x.

    Parameters
    ----------
    x : int, float, Dual

    Returns
    -------
    y : float or Dual

    Examples
    --------
    >>> ad.cosh(1)
    1.5430806348152437

    >>> x = ad.Dual(2, 1)
    >>> ad.cosh(x)
    Dual(3.7621956910836314, array([3.62686041]))
    """
    try:
        val = np.cosh(x.val)
        der = np.sinh(x.val) * x.der
        return Dual(val, der)
    except AttributeError:
        return np.cosh(x)


def tanh(x):
    """
    Return the hyperbolic tangent of x.

    Parameters
    ----------
    x : int, float, Dual

    Returns
    -------
    y : float or Dual

    Examples
    --------
    >>> ad.tanh(1)
    0.7615941559557649

    >>> x = ad.Dual(1, 1)
    >>> ad.tanh(x)
    Dual(0.7615941559557649, array([0.41997434]))
    """
    try:
        val = np.tanh(x.val)
        der = (1 - np.tanh(x.val)**2) * x.der
        return Dual(val, der)
    except AttributeError:
        return np.tanh(x)


def arcsin(x):
    """
    Return the inverse sine of x.

    Parameters
    ----------
    x : int, float, Dual

    Returns
    -------
    y : float or Dual

    Notes
    -----
    The derivative of arcsin(x) is undefined when x is not in the range (-1, 1).

    Examples
    --------
    >>> ad.arcsin(1)
    1.5707963267948966

    >>> x = ad.Dual(0.5, 1)
    >>> ad.arcsin(x)
    Dual(0.5235987755982988, array([1.15470054]))

    Derivative undefined:

    >>> x = ad.Dual(1, 1)
    >>> ad.arcsin(x)
    Traceback (most recent call last):
    ...
    ValueError: Derivative of arcsin(x) is undefined for x = 1
    """
    try:
        if abs(x.val) >= 1:
            raise ValueError(
                f"Derivative of arcsin(x) is undefined for x = {x.val}")
        val = np.arcsin(x.val)
        der = x.der / (np.sqrt(1 - x.val**2))
        return Dual(val, der)
    except AttributeError:
        return np.arcsin(x)


def arccos(x):
    """
    Return the inverse cosine of x.

    Parameters
    ----------
    x : int, float, Dual

    Returns
    -------
    y : float or Dual

    Notes
    -----
    The derivative of arccos(x) is undefined when x is not in the range (-1, 1).

    Examples
    --------
    >>> ad.arccos(1)
    0.0

    >>> x = ad.Dual(0.5, 1)
    >>> ad.arccos(x)
    Dual(1.0471975511965976, array([-1.15470054]))

    Derivative undefined:

    >>> x = ad.Dual(1, 1)
    >>> ad.arccos(x)
    Traceback (most recent call last):
    ...
    ValueError: Derivative of arccos(x) is undefined for x = 1
    """
    try:
        if abs(x.val) >= 1:
            raise ValueError(
                f"Derivative of arccos(x) is undefined for x = {x.val}")
        val = np.arccos(x.val)
        der = -x.der / (np.sqrt(1 - x.val**2))
        return Dual(val, der)
    except AttributeError:
        return np.arccos(x)


def arctan(x):
    """
    Return the inverse tangent of x.

    Parameters
    ----------
    x : int, float, Dual

    Returns
    -------
    y : float or Dual

    Examples
    --------
    >>> ad.arctan(1)
    0.7853981633974483

    >>> x = ad.Dual(1, 1)
    >>> ad.arctan(x)
    Dual(0.7853981633974483, array([0.5]))
    """
    try:
        val = np.arctan(x.val)
        der = x.der / (1 + x.val**2)
        return Dual(val, der)
    except AttributeError:
        return np.arctan(x)


def exp(x):
    """
    Return the exponential of x.

    Parameters
    ----------
    x : int, float, Dual

    Returns
    -------
    y : float or Dual

    Examples
    --------
    >>> ad.exp(1)
    2.718281828459045

    >>> x = ad.Dual(1, -2)
    >>> ad.exp(x)
    Dual(2.718281828459045, array([-5.43656366]))
    """
    try:
        val = np.exp(x.val)
        der = np.exp(x.val) * x.der
        return Dual(val, der)
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

    >>> x = ad.Dual(2, -1.5)
    >>> ad.log(x)
    Dual(0.6931471805599453, array([-0.75]))

    Logarithm with different base:

    >>> x = ad.Dual(2, -1.5)
    >>> log(x, base=10)
    Dual(0.30102999566398114, array([-0.32572086]))

    >>> x = ad.Dual(2, -1.5)
    >>> log(x, base=2)
    Dual(1.0, array([-1.08202128]))

    Derivative undefined:

    >>> x = ad.Dual(0, 1)
    >>> ad.log(x)
    Traceback (most recent call last):
    ...
    ValueError: Log of x is undefined for x = 0

    Error when base is non-positive:

    >>> x = ad.Dual(1, 1)
    >>> ad.log(x, base=0)
    Traceback (most recent call last):
    ...
    ValueError: Logarithm base must be positive
    """
    if base <= 0:
        raise ValueError(f"Logarithm base must be positive")
    try:
        if x.val <= 0:
            raise ValueError(f"Log of x is undefined for x = {x.val}")
        val = np.log(x.val) / np.log(base)
        der = x.der / (x.val * np.log(base))
        return Dual(val, der)
    except AttributeError:
        if x <= 0:
            raise ValueError(f"Log of x is undefined for x = {x}")
        return np.log(x) / np.log(base)


def sqrt(x):
    """
    Return the square root of x.

    Parameters
    ----------
    x : int, float, Dual

    Returns
    -------
    y : float or Dual

    Notes
    -----
    The square root of x is undefined when x is less than 0.

    Examples
    --------
    >>> ad.sqrt(4)
    2.0

    >>> x = ad.Dual(4, -1.5)
    >>> ad.sqrt(x)
    Dual(2.0, array([-0.375]))

    Derivative undefined:

    >>> x = ad.Dual(-1, 1)
    >>> ad.sqrt(x)
    Traceback (most recent call last):
    ...
    ValueError: sqrt(x) is undefined for x < 0
    """
    try:
        if x.val < 0:
            raise ValueError(f"sqrt(x) is undefined for x < 0")
        val = np.sqrt(x.val)
        der = (0.5 / val) * x.der
        return Dual(val, der)
    except AttributeError:
        if x < 0:
            raise ValueError(f"sqrt(x) is undefined for x < 0")
        return np.sqrt(x)


def logistic(x):
    """
    Return the logistic function of x.

    Parameters
    ----------
    x : int, float, Dual

    Returns
    -------
    y : float or Dual

    Examples
    --------
    >>> ad.logistic(1)
    0.7310585786300049

    >>> x = ad.Dual(3, 2)
    >>> ad.logistic(x)
    Dual(0.9525741268224334, array([0.09035332]))
    """
    g = lambda z: 1 / (1 + np.exp(-z))
    try:
        val = g(x.val)
        der = x.der * g(x.val) * (1 - g(x.val))
        return Dual(val, der)
    except AttributeError:
        return g(x)
