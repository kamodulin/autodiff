import numpy as np


class Dual:
    """
    Primary data structure for forward mode automatic differentiation.

    Dual numbers can be used as a data structure to store the value and
    derivative of a function. The value and derivative can be stored as
    the real and "dual" part of a dual number, respectively. The properties
    of a dual number lend itself nicely to a straightforward implementation of
    forward mode automatic differentiation.

    Parameters
    ----------
    val : float
        The value of the Dual number.
    der : ndarray
        The derivative of the Dual number.

    Examples
    --------
    Construct a Dual number for a univariate function:

    >>> x = ad.Dual(42)
    >>> x
    Dual(42, array([1]))

    Construct a Dual number for a multivariate function with user-defined seed vector:

    >>> x = ad.Dual(42, [1, 2])
    >>> x
    Dual(42, array([1, 2]))

    Construct multiple Dual numbers from array of scalars:

    >>> x, y, z = ad.Dual.from_array([1, 2, 4])
    >>> x
    Dual(1, array([1, 0, 0]))
    >>> y
    Dual(2, array([0, 1, 0]))

    Create a function from multiple Dual numbers:

    >>> x, y, z = ad.Dual.from_array([1, 2, 4])
    >>> f = (x * y)/z
    >>> f.val
    0.5
    >>> f.der
    array([0.5, 0.25, -0.125])

    See Also
    --------
    Dual.constant
    Dual.from_array

    """
    def __init__(self, val, der=1):
        self.val = val
        self.der = np.array(der, ndmin=1)

    @property
    def ndim(self):
        """
        Return the number of dimensions of the Dual number.

        Parameters
        ----------
        self : Dual

        Returns
        -------
        out : int
            Number of dimensions.
        
        Examples
        --------
        >>> ad.Dual(-5, [6.2]).ndim
        1
        
        More than one dimension:
        
        >>> ad.Dual(42, [1, 2]).ndim
        2
        """
        return len(self.der)

    @staticmethod
    def constant(val, ndim=1):
        """
        Create a Dual number representing a constant.

        Parameters
        ----------
        val : int or float
            Value of dual number.

        ndim : int, optional
            ``ndim`` is the number of dimensions of the zero derivative vector.

        Returns
        -------
        out : Dual
            Dual number of value ``val`` with zero derivative vector.

        Notes
        -----
        Derivative vector of length ``ndim`` will be filled with zeros.

        Examples
        --------
        >>> ad.Dual.constant(42)
        Dual(42, array([0.]))

        Constant with more than one dimension:

        >>> ad.Dual.constant(7, 2)
        Dual(7, array([0., 0.]))
        """
        zeros = np.zeros(ndim)
        return Dual(val, zeros)

    @staticmethod
    def from_array(X):
        """
        Generate Dual numbers for a multivariable function.

        Parameters
        ----------
        X : array
            Array of numbers which will be values of Dual numbers.

        Returns
        -------
        out : Dual number generator
            Dual numbers of value ``X[i]`` with zero derivative vector
            where the i-th element has a value of 1.

        Examples
        --------
        >>> x, y = ad.Dual.from_array([1, 42])
        >>> x
        Dual(1, array([1., 0.]))
        >>> y
        Dual(42, array([0., 1.]))
        """
        if np.ndim(X) != 1:
            raise Exception(f"array must be 1-dimensional")
        if len(X) == 1:
            return Dual(X[0], 1)

        I = np.identity(len(X))
        return iter(Dual(x, I[i]) for i, x in enumerate(X))

    def _compatible(self, other, operand=None):
        """
        Return other element if compatible with type ``Dual`` and ensure that the
        number of dimensions match between the two Duals.

        Parameters
        ----------
        self : Dual
        other : Dual
        operand : str, optional

        Returns
        -------
        out : Dual, ArithmeticError, or TypeError
            Dual if ``other`` is compatible. Raise error if other does not have
            matching dimensions or not of correct type.

        Examples
        --------
        >>> d0 = ad.Dual(42)
        >>> d1 = ad.Dual(1)
        >>> d0._compatible(d1)
        Dual(1, array([1]))

        Dimension mismatch:

        >>> d2 = ad.Dual(10, [0, 1])
        >>> d0._compatible(d2)
        Traceback (most recent call last):
        ...
        ArithmeticError: Dimensionality mismatch between Dual(42, array([1])) and Dual(10, array([0, 1]))

        Incorrect type:

        >>> d0._compatible("autodiff")
        Traceback (most recent call last):
        ...
        TypeError: unsupported operand type(s) for None: 'Dual' and 'str'
        """
        if isinstance(other, (int, float)):
            return Dual.constant(other, ndim=self.ndim)
        elif isinstance(other, Dual):
            if self.ndim == other.ndim:
                return other
            raise ArithmeticError(
                f"Dimensionality mismatch between {self} and {other}")
        raise TypeError(
            f"unsupported operand type(s) for {operand}: '{type(self).__name__}' and '{type(other).__name__}'"
        )

    def __repr__(self):
        """
        Return a string representation of the Dual number.

        Parameters
        ----------
        self : Dual

        Returns
        -------
        out : str
        """
        return f"{self.__class__.__name__}({self.val}, {np.array_repr(self.der)})"

    def __add__(self, other):
        """
        Return the sum of ``self`` and ``other``.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual
        
        Returns
        -------
        out : Dual
        
        Examples
        --------
        Dual number and scalar:

        >>> ad.Dual(42) + 5
        Dual(47, array([1]))

        Two dual numbers (univariate):

        >>> ad.Dual(42) + ad.Dual(1)
        Dual(43, array([2]))

        Two dual numbers (multivariate):

        >>> ad.Dual(42, [1, 2]) + ad.Dual(1, [3, 4])
        Dual(43, array([4, 6]))
        """
        if other := self._compatible(other, "+"):
            return Dual(self.val + other.val, self.der + other.der)

    def __radd__(self, other):
        """
        Return the sum of two numbers, when the left operand is not a Dual
        number.

        Parameters
        ----------
        self : Dual
        other : int, float

        Returns
        -------
        out : Dual
        
        Examples
        --------
        Scalar and dual number (univariate):

        >>> 1.2 + ad.Dual(42)
        Dual(43.2, array([1]))

        Scalar and dual number (multivariate):

        >>> -3.6 + ad.Dual(42, [1, 2])
        Dual(38.4, array([1, 2]))
        """
        return self + other

    def __sub__(self, other):
        """
        Returns the difference between ``self`` and ``other``.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual

        Returns
        -------
        out : Dual

        Examples
        --------
        Dual number and scalar:

        >>> ad.Dual(42) - 5
        Dual(37, array([1]))

        Two dual numbers (univariate):

        >>> ad.Dual(42) - ad.Dual(1)
        Dual(41, array([0]))

        Two dual numbers (multivariate):

        >>> ad.Dual(42, [1, 2]) - ad.Dual(1, [3, 4])
        Dual(41, array([-2, -2]))
        """
        if other := self._compatible(other, "-"):
            return Dual(self.val - other.val, self.der - other.der)

    def __rsub__(self, other):
        """
        Return the difference between two numbers, when the left operand is not
        a Dual number.
        
        Parameters
        ----------
        self : Dual
        other : int, float

        Returns
        -------
        out : Dual

        Examples
        --------
        Scalar and dual number (univariate):

        >>> 1.2 - ad.Dual(42)
        Dual(-40.8, array([-1.]))

        Scalar and dual number (multivariate):

        >>> -3.6 - ad.Dual(42, [1, 2])
        Dual(-45.6, array([-1., -2.]))
        """
        if other := self._compatible(other, "-"):
            return Dual(other.val - self.val, other.der - self.der)

    def __mul__(self, other):
        """
        Return the product of ``self`` and ``other``.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual

        Returns
        -------
        out : Dual

        Examples
        --------
        Dual number and scalar:

        >>> ad.Dual(42) * 5
        Dual(210, array([5]))

        Two dual numbers (univariate):

        >>> ad.Dual(5.6) * ad.Dual(1)
        Dual(5.6, array([6.6]))

        Two dual numbers (multivariate):

        >>> ad.Dual(-9, [1, 2]) * ad.Dual(4, [2, -9])
        Dual(-36, array([-14,  89]))
        """
        if other := self._compatible(other, "*"):
            return Dual(self.val * other.val,
                        self.val * other.der + self.der * other.val)

    def __rmul__(self, other):
        """
        Return the product of two numbers, when the left operand is not a Dual
        number.

        Parameters
        ----------
        self : Dual
        other : int, float

        Returns
        -------
        out : Dual

        Examples
        --------
        Scalar and dual number (univariate):

        >>> 1.2 * ad.Dual(42)
        Dual(50.4, array([1.2]))

        Scalar and dual number (multivariate):

        >>> -3 * ad.Dual(42, [1, 2])
        Dual(-126, array([-3., -6.]))
        """
        return self * other

    def __truediv__(self, other):
        """
        Return the quotient of ``self`` and ``other``.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual

        Returns
        -------
        out : Dual

        Examples
        --------
        Dual number and scalar:

        >>> ad.Dual(42) / 5
        Dual(8.4, array([0.2]))

        Two dual numbers (univariate):

        >>> ad.Dual(4) / ad.Dual(5)
        Dual(0.8, array([0.04]))

        Two dual numbers (multivariate):

        >>> ad.Dual(42, [1, 2]) / ad.Dual(1, [3, 4])
        Dual(42.0, array([-125., -166.]))
        """
        if other := self._compatible(other, "/"):
            return Dual(self.val / other.val,
                        (other.val * self.der - self.val * other.der) /
                        (other.val**2))

    def __rtruediv__(self, other):
        """
        Return the quotient of two numbers, when the left operand is not a Dual
        number.

        Parameters
        ----------
        self : Dual
        other : int, float

        Returns
        -------
        out : Dual

        Examples
        --------
        Scalar and dual number (univariate):

        >>> 2 / ad.Dual(4)
        Dual(0.5, array([-0.125]))

        Scalar and dual number (multivariate):

        >>> 2 / ad.Dual(4, [1, 2])
        Dual(0.5, array([-0.125, -0.25 ]))
        """
        if other := self._compatible(other, "/"):
            return Dual(other.val / self.val,
                        (self.val * other.der - other.val * self.der) /
                        (self.val**2))

    def __pow__(self, other):
        """
        Return ``self`` to the power of ``other``.

        Parameters
        ----------
        self : Dual
        other : int, float

        Returns
        -------
        out : Dual

        Notes
        -----
        If ``other`` is a non-integer scalar and ``self.val`` is less than zero,
        the derivative is a complex number. This will raise a ValueError. Only integer
        powers are supported if the base is negative.

        If ``self.val`` is equal to zero and ``other`` is less than one, we cannot
        compute the derivative of the result due to a ZeroDivisionError.
        
        If ``other`` is a Dual and ``self.val`` is less than or equal to zero, we cannot
        compute the derivative of the result since the log of a negative number is
        not defined. This will raise a ValueError.

        Examples
        --------
        Dual number and scalar:

        >>> ad.Dual(2) ** 5
        Dual(32, array([80]))

        Two dual numbers (univariate):

        >>> ad.Dual(2, [1]) ** ad.Dual(3, [2])
        Dual(8, array([23.09035489]))

        Two dual numbers (multivariate):

        >>> ad.Dual(3, [1, 2]) ** ad.Dual(2, [3, 4])
        Dual(9, array([35.66253179, 51.55004239]))

        Example of errors raised (see notes above):

        >>> ad.Dual(-1) ** 1.2
        Traceback (most recent call last):
        ...
        ValueError: -1 cannot be raised to the power of 1.2; only integer powers are allowed if base is negative
        
        >>> ad.Dual(0) ** -2
        Traceback (most recent call last):
        ...
        ZeroDivisionError: 0.0 cannot be raised to a negative power
        
        >>> ad.Dual(0) ** ad.Dual(1)
        Traceback (most recent call last):
        ...
        ValueError: 0 cannot be raised to the power of 1; log is undefined for x = 0
        """
        if isinstance(other, (int, float)):
            if self.val < 0 and (other != int(other)):
                raise ValueError(
                    f"{self.val} cannot be raised to the power of {other}; only integer powers are allowed if base is negative"
                )
            elif self.val == 0 and other < 1:
                raise ZeroDivisionError(
                    f"0.0 cannot be raised to a negative power")
        elif isinstance(other, Dual):
            if self.val <= 0:
                raise ValueError(
                    f"{self.val} cannot be raised to the power of {other.val}; log is undefined for x = {self.val}"
                )
        try:
            der_comp_2 = other.der * np.log(
                self.val) + other.val * (self.der / self.val)
            return Dual(self.val**other.val,
                        (self.val**other.val) * der_comp_2)
        except AttributeError:
            return Dual(self.val**other,
                        other * self.val**(other - 1) * self.der)

    def __rpow__(self, other):
        """
        Return ``other`` to the power of ``self`` if ``other`` is not a Dual number.

        Parameters
        ----------
        self : Dual
        other : int, float

        Returns
        -------
        out : Dual

        Examples
        --------
        Scalar and dual number (univariate):

        >>> 5 ** ad.Dual(2)
        Dual(25, array([40.23594781]))

        Scalar and dual number (multivariate):

        >>> ad.Dual(2, [1]) ** 3
        Dual(8, array([12]))

        Two dual numbers (univariate):

        >>> ad.Dual(2, [1, 2]) ** ad.Dual(3, [3, 4])
        Dual(8, array([28.63553233, 46.18070978]))
        """
        if other <= 0:
            raise ValueError(
                f"{other} cannot be raised to the power of {self.val}; log is undefined for x = {other}"
            )

        val = other**self.val
        der = val * np.log(other) * self.der
        return Dual(val, der)

    def __neg__(self):
        """
        Return negation of ``self``.

        Parameters
        ----------
        self : Dual

        Returns
        -------
        out : Dual

        Examples
        --------
        Dual number (univariate):

        >>> -ad.Dual(42)
        Dual(-42, array([-1]))

        Dual number (multivariate):

        >>> -ad.Dual(42, [1, 2])
        Dual(-42, array([-1, -2]))
        """
        return Dual(-self.val, -self.der)

    def __lt__(self, other):
        """
        Return element-wise (value and derivative vector) less than comparison of
        ``self`` and ``other``.
        and ``other``.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual

        Returns
        -------
        out : tuple

        Examples
        --------
        Dual number and scalar:

        >>> ad.Dual(42) < 5
        (False, array([False]))
        
        Two dual numbers (univariate):

        >>> ad.Dual(1, [2]) < ad.Dual(5, [1])
        (True, array([False]))

        Two dual numbers (multivariate):

        >>> ad.Dual(1, [4, 1]) < ad.Dual(5, [1, 2])
        (True, array([False, True]))
        """
        if other := self._compatible(other, "<"):
            return self.val < other.val, self.der < other.der

    def __gt__(self, other):
        """
        Return element-wise (value and derivative vector) greater than comparison of
        ``self`` and ``other``.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual

        Returns
        -------
        out : tuple

        Examples
        --------
        Dual number and scalar:

        >>> ad.Dual(42) > 5
        (True, array([True]))

        Two dual numbers (univariate):

        >>> ad.Dual(1, [2]) > ad.Dual(5, [1])
        (False, array([True]))

        Two dual numbers (multivariate):

        >>> ad.Dual(1, [4, 1]) > ad.Dual(5, [1, 2])
        (False, array([True, False]))
        """
        if other := self._compatible(other, ">"):
            return self.val > other.val, self.der > other.der

    def __le__(self, other):
        """
        Return element-wise (value and derivative vector) less than or equal to
        comparison of ``self`` and ``other``.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual

        Returns
        -------
        out : tuple

         Examples
        --------
        Dual number and scalar:

        >>> ad.Dual(42) <= 5
        (False, array([False]))

        Two dual numbers (univariate):

        >>> ad.Dual(1, [2]) <= ad.Dual(5, [2])
        (True, array([True]))

        Two dual numbers (multivariate):

        >>> ad.Dual(6, [1, 1]) <= ad.Dual(5, [1, 2])
        (False, array([True, True]))
        """
        if other := self._compatible(other, "<="):
            return self.val <= other.val, self.der <= other.der

    def __ge__(self, other):
        """
        Return element-wise (value and derivative vector) greater than or equal to
        comparison of ``self`` and ``other``.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual

        Returns
        -------
        out : tuple

        Examples
        --------
        Dual number and scalar:

        >>> ad.Dual(42) >= 5
        (True, array([True]))

        Two dual numbers (univariate):

        >>> ad.Dual(5, [2]) >= ad.Dual(5, [1])
        (True, array([True]))

        Two dual numbers (multivariate):

        >>> ad.Dual(5, [4, 2]) >= ad.Dual(5, [1, 2])
        (True, array([True, True]))
        """
        if other := self._compatible(other, ">="):
            return self.val >= other.val, self.der >= other.der

    def __eq__(self, other):
        """
        Return element-wise (value and derivative vector) equality comparison of
        ``self`` and ``other``.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual

        Returns
        -------
        out : tuple

        Examples
        --------
        Dual number and scalar:

        >>> ad.Dual(42) == 5
        (False, array([False]))

        Two dual numbers (univariate):

        >>> ad.Dual(5, [2]) == ad.Dual(5, [1])
        (True, array([False]))

        Two dual numbers (multivariate):

        >>> ad.Dual(2, [4, 2]) == ad.Dual(5, [1, 2])
        (False, array([False,  True]))
        """
        if other := self._compatible(other, "=="):
            return self.val == other.val, self.der == other.der

    def __ne__(self, other):
        """
        Return element-wise (value and derivative vector) inequality comparison of
        ``self`` and ``other``.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual

        Returns
        -------
        out : tuple

        Examples
        --------
        Dual number and scalar:

        >>> ad.Dual(42) != 5
        (True, array([True]))

        Two dual numbers (univariate):

        >>> ad.Dual(5, [2]) != ad.Dual(5, [1])
        (False, array([True]))

        Two dual numbers (multivariate):

        >>> ad.Dual(2, [4, 2]) != ad.Dual(5, [1, 2])
        (True, array([True,  False]))
        """
        if other := self._compatible(other, "!="):
            return self.val != other.val, self.der != other.der
