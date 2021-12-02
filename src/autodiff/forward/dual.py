import numpy as np


class Dual:
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
        >>> Dual(42, [1, 2]).ndim
        2
        >>> Dual(-5, [6.2]).ndim
        1
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
            `ndim` is the number of dimensions of the zero derivative vector.

        Returns
        -------
        out : Dual
            Dual number of value `val` with zero derivative vector.

        Notes
        -----
        Derivative vector of length `ndim` will be filled with zeros.

        Examples
        --------
        >>> Dual.constant(42)
        Dual(42, array([0.]))
        >>> Dual.constant(7, 2)
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
            Dual numbers of value `X[i]` with zero derivative vector
            where the i-th element has a value of 1.

        Examples
        --------
        >>> x, y = Dual.from_array([1, 42])
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
        Return other element if compatible with type `Dual` and ensure that the
        number of dimensions match between the two Duals.

        Parameters
        ----------
        self : Dual
        other : Dual
        operand : str, optional

        Returns
        -------
        out : Dual, ArithmeticError, or TypeError
            Dual if `other` is compatible. Raise error if other does not have
            matching dimensions or not of correct type.

        Examples
        --------
        >>> d0 = Dual(42)
        >>> d0._compatible(Dual(1))
        Dual(1, array([1]))
        >>> d0._compatible(Dual(10, [0, 1]))
        Traceback (most recent call last):
        ...
        ArithmeticError: Dimensionality mismatch between Dual(42, array([1])) and Dual(10, array([0, 1]))
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
        Return the sum of `self` and `other`.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual
        
        Returns
        -------
        out : Dual
        
        Examples
        --------
        >>> Dual(42) + 5
        Dual(47, array([1]))
        >>> Dual(42) + Dual(1)
        Dual(43, array([2]))
        >>> Dual(42, [1, 2]) + Dual(1, [3, 4])
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
        >>> 1.2 + Dual(42)
        Dual(43.2, array([1]))
        >>> -3.6 + Dual(42, [1, 2])
        Dual(38.4, array([1, 2]))
        """
        return self + other

    def __sub__(self, other):
        """
        Returns the difference between `self` and `other`.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual

        Returns
        -------
        out : Dual

        Examples
        --------
        >>> Dual(42) - 5
        Dual(37, array([1]))
        >>> Dual(42) - Dual(1)
        Dual(41, array([0]))
        >>> Dual(42, [1, 2]) - Dual(1, [3, 4])
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
        >>> 1.2 - Dual(42)
        Dual(-40.8, array([-1.]))
        >>> -3.6 - Dual(42, [1, 2])
        Dual(-45.6, array([-1., -2.]))
        """
        if other := self._compatible(other, "-"):
            return Dual(other.val - self.val, other.der - self.der)

    def __mul__(self, other):
        """
        Return the product of `self` and `other`.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual

        Returns
        -------
        out : Dual

        Examples
        --------
        >>> Dual(42) * 5
        Dual(210, array([5]))
        >>> Dual(5.6) * Dual(1)
        Dual(5.6, array([6.6]))
        >>> Dual(-9, [1, 2]) * Dual(4, [2, -9])
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
        >>> 1.2 * Dual(42)
        Dual(50.4, array([1.2]))
        >>> -3 * Dual(42, [1, 2])
        Dual(-126, array([-3., -6.]))
        """
        return self * other

    def __truediv__(self, other):
        """
        Return the quotient of `self` and `other`.

        Parameters
        ----------
        self : Dual
        other : int, float, Dual

        Returns
        -------
        out : Dual

        Examples
        --------
        >>> Dual(42) / 5
        Dual(8.4, array([0.2]))
        >>> Dual(4) / Dual(5)
        Dual(0.8, array([0.04]))
        >>> Dual(42, [1, 2]) / Dual(1, [3, 4])
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
        >>> 2 / Dual(4)
        Dual(0.5, array([-0.125]))
        >>> 2 / Dual(4, [1, 2])
        Dual(0.5, array([-0.125, -0.25 ]))
        """
        if other := self._compatible(other, "/"):
            return Dual(other.val / self.val,
                        (self.val * other.der - other.val * self.der) /
                        (self.val**2))

    def __pow__(self, other):
        """
        Return `self` to the power of `other`.

        Parameters
        ----------
        self : Dual
        other : int, float

        Returns
        -------
        out : Dual

        Examples
        --------
        >>> Dual(2) ** 5
        Dual(32, array([80.]))
        >>> Dual(2, [1]) ** Dual(3, [2])
        Dual(8, array([23.09035489]))
        >>> Dual(3, [1, 2]) ** Dual(2, [3, 4])
        Dual(9, array([35.66253179, 51.55004239]))
        """
        if isinstance(other, (int, float)):
            if self.val < 0 and (other != int(other)): # complex result
                raise ValueError(f"{self.val} cannot be raised to the power of {other}; only integer powers are allowed if base is negative")
            elif self.val == 0 and other < 1:
                raise ZeroDivisionError(f"0.0 cannot be raised to a negative power")
        elif isinstance(other, Dual):
            if self.val <= 0: # cannot take log of negative number
                raise ValueError(f"{self.val} cannot be raised to the power of {other.val}; log is undefined for x = {self.val}")
        try:
            der_comp_2 = other.der * np.log(
                self.val) + other.val * (self.der / self.val)
            return Dual(self.val ** other.val,
                        (self.val ** other.val) * der_comp_2)
        except AttributeError:
            return Dual(self.val ** other, other * self.val ** (other - 1) * self.der)

    def __rpow__(self, other):
        if other <= 0:
            raise ValueError(f"{other} cannot be raised to the power of {self.val}; log is undefined for x = {other}")
        
        val = other ** self.val
        der = val * np.log(other) * self.der
        return Dual(val, der)

    def __neg__(self):
        return Dual(-self.val, -self.der)

    def __lt__(self, other):
        if other := self._compatible(other, "<"):
            return self.val < other.val, self.der < other.der

    def __gt__(self, other):
        if other := self._compatible(other, ">"):
            return self.val > other.val, self.der > other.der

    def __le__(self, other):
        if other := self._compatible(other, "<="):
            return self.val <= other.val, self.der <= other.der

    def __ge__(self, other):
        if other := self._compatible(other, ">="):
            return self.val >= other.val, self.der >= other.der

    def __eq__(self, other):
        if other := self._compatible(other, "=="):
            return self.val == other.val, self.der == other.der

    def __ne__(self, other):
        if other := self._compatible(other, "!="):
            return self.val != other.val, self.der != other.der
