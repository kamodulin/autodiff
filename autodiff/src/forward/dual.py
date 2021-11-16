import numpy as np


class Dual:
    def __init__(self, val, der=1):
        self.val = val
        self.der = np.array(der, ndmin=1)

    @property
    def ndim(self):
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
        I = np.identity(len(X))
        return iter(Dual(x, I[i]) for i, x in enumerate(X))

    def _compatible(self, other, operand=None):
        """
        Return other element if compatible with type `Dual` and ensure that the
        number of dimensions match between the two Duals.

        Parameters
        ----------
        other : Dual
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
        return f"{self.__class__.__name__}({self.val}, {np.array_repr(self.der)})"

    def __add__(self, other):
        if other := self._compatible(other, "+"):
            return Dual(self.val + other.val, self.der + other.der)

    def __radd__(self, other):
        return self + other

    def __sub__(self):
        ...

    def __rsub__(self):
        ...

    def __mul__(self, other):
        if other := self._compatible(other, "*"):
            return Dual(self.val * other.val,
                        self.val * other.der + self.der * other.val)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self):
        ...

    def __rtruediv__(self):
        ...

    def __pow__(self, other):
        if other := self._compatible(other, "**"):
            der_comp_2 = other.der*np.log(self.val) + other.val*(self.der/self.val)
            return Dual(self.val ** other.val, (self.val ** other.val)*der_comp_2)

    def __rpow__(self, other):
        if other := self._compatible(other, "**"):
            return other ** self

    def __neg__(self):
        ...

    def __lt__(self):
        ...

    def __gt__(self):
        ...

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
