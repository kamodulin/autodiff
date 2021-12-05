import numpy as np
import warnings

class Node:
    def __init__(self, val):
        self.val = val
        self.children = []
        self.der = None

    def grad(self):
        """
        Return the gradient of the last descendent with respect to self 

        Parameters
        ----------
        self: Node

        Returns
        -------
        out: float

        Examples
        --------
        >>> x, y = Node(1),Node(1)
        >>> x + 3*y
        Node(4)
        >>> x.grad()
        1.0
        >>> y.grad()
        3.0
        """
        if self.children is not None and len(self.children) == 0:
            return 1.0
        if self.der is None:
            self.der = sum(w*node.grad() for w, node in self.children)
        return self.der

    @staticmethod
    def zero_grad(*args):
        """
        Reset Nodes to their default attributes

        Parameters
        ----------
        *args: arbitrary number of Nodes

        Returns
        --------
        None
        
        Examples
        --------
        >>> x = Node(3)
        >>> f = ad.sin(x)
        >>> x.grad()
        -0.9899924966004454
        >>> Node.zero_grad(x)
        >>> x.grad()
        1.0
        """
        for arg in args:
            try:
                arg.children = []
                arg.der = None
            except AttributeError:
                raise AttributeError(f'Cannot set gradient to zero for type {arg.__class__.__name__}')

    @staticmethod
    def constant(val):
        """
        Create a Node representing a constant.

        Parameters
        ----------
        val : int or float
            Value of Node.

        Returns
        -------
        out : Node
            Node of value `val` with children set to None and der set to 1.0.

        Examples
        --------
        >>> Node.constant(42)
        Node(42)
        >>> Node.constant(1)
        Node(1)
        """
        node = Node(val)
        node.children = None
        node.der = 0
        return node

    @staticmethod
    def from_array(X):
        """
        Generate Nodes for a multivariable function.
        
        Parameters
        ----------
        X : array
            Array of numbers which will be values of Nodes.

        Returns
        -------
        out : Nodes generator
            Nodes of value `X[i]` with der set to None
            and an empty list of children.
        
        Examples
        --------
        >>> x, y, z = Node.from_array([1,2,3])
        >>> x
        Node(1)
        >>> y
        Node(2)
        >>> z
        Node(3)
        """
        if np.ndim(X) != 1:
            raise Exception(f"array must be 1-dimensional")
        if len(X) == 1:
            return Node(X[0])

        return iter(Node(x) for x in X)

    def _isConstant(self, other, operand = None):
        """
        Return other element as a constant Node if other is a number and raises
        an type error if other is neither a number nor a Node

        Parameters
        ----------
        self : Node
        other : Node
        operand : str, optional

        Returns
        -------
        out : Node or TypeError
            Node if `other` is a float, int or Node. Raise error if other is neither
            Node nor a number

        Examples
        --------
        >>> x = Node(42)
        >>> x._isConstant(Node(1))
        Node(1)
        >>> x._isConstant(1)
        Node(1)
        >>> x._isConstant("autodiff")
        Traceback (most recent call last):
        ...
        TypeError: unsupported operand type(s) for None: 'Node' and 'str'
        """
        if isinstance(other, (int, float)):
            return Node.constant(other)
        elif isinstance(other, Node):
            return other
        raise TypeError(f"unsupported operand type(s) for {operand}: '{type(self).__name__}' and '{type(other).__name__}'")


    def _addChildren(self, new_weight, new_child):
        """
        Add a tuple of (new_weight, new _child) to self's children list if children
        list is not None

        Parameters
        ----------
        self : Node
        new_weight : float or int
        new_child : Node

        Returns
        -------
        None

        Examples
        --------
        >>> x = Node(42)
        >>> x._addChildren(4.2,Node(1))
        >>> x.children
        [(4.2, Node(1))]
        """
        if self.children is not None:
            self.children.append((new_weight, new_child))

    def __add__(self,other):
        """
        Return the sum of `self` and `other`.

        Parameters
        ----------
        self : Node
        other : int, float, Node
        
        Returns
        -------
        out : Node
        
        Examples
        --------
        >>> Node(42) + 5
        Node(47)
        >>> Node(42) + Node(1)
        Node(43)
        >>> Node(42) + Node.constant(1)
        Node(43)
        """
        if other := self._isConstant(other):
            child = Node(self.val + other.val)
            self._addChildren(1.0,child)
            other._addChildren(1.0,child)
            return child

    def __radd__(self,other):
        """
        Return the sum of two numbers, when the left operand is not a Node

        Parameters
        ----------
        self : Node
        other : int, float

        Returns
        -------
        out : Node
        
        Examples
        --------
        >>> 1.2 + Node(42)
        Node(43.2)
        >>> -3.6 + Node.constant(42)
        Node(38.4)
        """
        return self + other

    def __mul__(self,other):
        """
        Return the product of `self` and `other`.

        Parameters
        ----------
        self : Node
        other : int, float, Node

        Returns
        -------
        out : Node

        Examples
        --------
        >>> Node(42) * 5
        Node(210)
        >>> Node(5.6) * Node(1)
        Node(5.6)
        >>> Node.constant(-9) * Node(4)
        Node(-36)
        """
        if other := self._isConstant(other):
            child = Node(self.val*other.val)
            self._addChildren(other.val, child)
            other._addChildren(self.val, child)
            return child

    def __rmul__(self,other):
        """
        Return the product of two numbers, when the left operand is not a Node

        Parameters
        ----------
        self : Node
        other : int, float

        Returns
        -------
        out : Node

        Examples
        --------
        >>> 1.2 * Node(42)
        Node(50.4)
        >>> -3 * Node.constant(42)
        Node(-126)
        """
        return self * other

    def __sub__(self,other):
        """
        Returns the difference between `self` and `other`.

        Parameters
        ----------
        self : Node
        other : int, float, Node

        Returns
        -------
        out : Node

        Examples
        --------
        >>> Node(42) - 5
        Node(37)
        >>> Node(42) - Node(1)
        Node(41)
        >>> Node(42) - Node.constant(2)
        Node(40)
        """
        if other := self._isConstant(other):
            child = Node(self.val - other.val)
            self._addChildren(1.0,child)
            other._addChildren(-1.0,child)
            return child

    def __rsub__(self,other):
        """
        Return the difference between two numbers, when the left operand is not
        Node
        
        Parameters
        ----------
        self : Node
        other : int, float

        Returns
        -------
        out : Node

        Examples
        --------
        >>> 1.2 - Node(42)
        Node(-40.8)
        >>> -3.6 - Node.constant(42)
        Node(-45.6)
        """
        if other := self._isConstant(other):
            child = Node(other.val - self.val)
            self._addChildren(-1.0,child)
            other._addChildren(1.0,child)
            return child

    def __truediv__(self, other):
        """
        Return the quotient of `self` and `other`.

        Parameters
        ----------
        self : Node
        other : int, float, Node

        Returns
        -------
        out : Node

        Examples
        --------
        >>> Node(42) / 5
        Node(8.4)
        >>> Node(4) / Node(5)
        Node(0.8)
        >>> Node.constant(42) / Node(1)
        Node(42.0)
        """
        if other := self._isConstant(other):
            child = Node(self.val/other.val)
            self._addChildren(1/other.val,child)
            other._addChildren(-self.val/(other.val**2),child)
            return child

    def __rtruediv__(self, other):
        """
        Return the quotient of two numbers, when the left operand is not a Node

        Parameters
        ----------
        self : Node
        other : int, float

        Returns
        -------
        out : Node

        Examples
        --------        
        >>> 2 / Node(4)
        Node(0.5)
        >>> 2 / Node.constant(4)
        Node(0.5)
        """
        if other := self._isConstant(other):
            child = Node(other.val/self.val)
            self._addChildren(-other.val/(self.val**2),child)
            other._addChildren(1/self.val,child)
            return child
    
    def __pow__(self, other):
        """
        Return `self` to the power of `other`.

        Parameters
        ----------
        self : Node
        other : int, float

        Returns
        -------
        out : Node

        Notes
        -----
        If `other` is a non-integer scalar and `self.val` is less than zero,
        the derivative is a complex number. This will raise a ValueError. Only integer
        powers are supported if the base is negative.

        If `self.val` is equal to zero and `other` is less than one, we cannot
        compute the derivative of the result due to a ZeroDivisionError.
        
        If `other` is a Node and `self.val` is less than or equal to zero, we cannot
        compute the derivative of the result since the log of a negative number is
        not defined. This will raise a ValueError.

        Examples
        --------
        >>> Node(2) ** 5
        Node(32)
        >>> Node(2) ** Node(3)
        Node(8)
        >>> Node(2) ** Node.constant(3)
        Node(8)

        Errors
        ------
        >>> Node(-1) ** 1.2
        Traceback (most recent call last):
        ...
        ValueError: -1 cannot be raised to the power of 1.2; only integer powers are allowed if base is negative
        >>> Node(0) ** -2
        Traceback (most recent call last):
        ...
        ZeroDivisionError: 0.0 cannot be raised to a negative power
        >>> Node(0) ** Node(1)
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
        elif isinstance(other, Node):
            if self.val <= 0:
                raise ValueError(
                    f"{self.val} cannot be raised to the power of {other.val}; log is undefined for x = {self.val}"
                )
        try:
            val = self.val**other.val
            child = Node(val)
            self._addChildren(val * other.val / self.val, child)
            other._addChildren(val * np.log(self.val), child)
            return child
        except AttributeError:
            child = Node(self.val**other)
            self._addChildren(other*self.val**(other-1),child)
            return child

    def __rpow__(self, other):
        """
        Return `other` to the power of `self` if `other` is not a Node.

        Parameters
        ----------
        self : Node
        other : int, float

        Returns
        -------
        out : Node

        Examples
        --------
        >>> 5 ** Node(2)
        Node(25)
        >>> Node(2) ** 3
        Node(8)
        >>> Node(2) ** Node.constant(3)
        Node(8)
        """
        if other <= 0:
            raise ValueError(
                f"{other} cannot be raised to the power of {self.val}; log is undefined for x = {other}"
            )
        val = other**self.val
        child = Node(val)
        self._addChildren(val*np.log(other),child)
        return child

    def __neg__(self):
        """
        Return negation of `self`.

        Parameters
        ----------
        self : Node

        Returns
        -------
        out : Node

        Examples
        --------
        >>> -Node(42)
        Node(-42)
        >>> -Node.constant(42)
        Node(-42)
        """
        if self.children == None:
            child = Node.constant(-self.val)
        else:
            child = Node(-self.val)
            self._addChildren(-1.0,child)
        return child

    def __lt__(self, other):
        """
        Return element-wise (value and derivative) less than comparison of
        `self` and `other`.

        Parameters
        ----------
        self : Node
        other : int, float, Node

        Returns
        -------
        out : tuple

        Notes
        -----
        If either `other` or self has None der. We return None for the derivative comparison and 
        give user a warning indicating that they are attempting to compare two Nodes before their
        derivatives are computed

        Examples
        --------
        >>> Node(42) < 5
        RuntimeWarning: Attempting to compare two nodes with None derivatives
        warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
        (False, None)
        >>> Node.constant(42) < Node.constant(5)
        (False, False)
        >>> Node.constant(42) < Node.constant(50)
        (True, False)
        """
        if other := self._isConstant(other):
            der_cp = None
            if other.der is None or self.der is None:
                warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
            else:
                der_cp = self.der < other.der
            return self.val < other.val, der_cp

    def __gt__(self, other):
        """
        Return element-wise (value and derivative) greater than comparison of
        `self` and `other`.

        Parameters
        ----------
        self : Node
        other : int, float, Node

        Returns
        -------
        out : tuple

        Notes
        -----
        If either `other` or self has None der. We return None for the derivative comparison and 
        give user a warning indicating that they are attempting to compare two Nodes before their
        derivatives are computed

        Examples
        --------
        >>> Node(42) > 5
        RuntimeWarning: Attempting to compare two nodes with None derivatives
        warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
        (True, None)
        >>> Node.constant(42) > Node.constant(5)
        (True, False)
        >>> Node.constant(42) > Node.constant(50)
        (False, False)
        """
        if other := self._isConstant(other):
            der_cp = None
            if other.der is None or self.der is None:
                warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
            else:
                der_cp = self.der > other.der
            return self.val > other.val, der_cp

    def __le__(self, other):
        """
        Return element-wise (value and derivative) less than or equal comparison of
        `self` and `other`.

        Parameters
        ----------
        self : Node
        other : int, float, Node

        Returns
        -------
        out : tuple

        Notes
        -----
        If either `other` or self has None der. We return None for the derivative comparison and 
        give user a warning indicating that they are attempting to compare two Nodes before their
        derivatives are computed

        Examples
        --------
        >>> Node(42) <= 5
        RuntimeWarning: Attempting to compare two nodes with None derivatives
        warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
        (False, None)
        >>> Node.constant(42) <= Node.constant(5)
        (False, True)
        >>> Node.constant(42) <= Node.constant(50)
        (True, True)
        """
        if other := self._isConstant(other):
            der_cp = None
            if other.der is None or self.der is None:
                warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
            else:
                der_cp = self.der <= other.der
            return self.val <= other.val, der_cp

    def __ge__(self, other):
        """
        Return element-wise (value and derivative) greater than or equal comparison of
        `self` and `other`.

        Parameters
        ----------
        self : Node
        other : int, float, Node

        Returns
        -------
        out : tuple

        Notes
        -----
        If either `other` or self has None der. We return None for the derivative comparison and 
        give user a warning indicating that they are attempting to compare two Nodes before their
        derivatives are computed

        Examples
        --------
        >>> Node(42) >= 5
        RuntimeWarning: Attempting to compare two nodes with None derivatives
        warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
        (True, None)
        >>> Node.constant(42) >= Node.constant(5)
        (True, True)
        >>> Node.constant(42) >= Node.constant(50)
        (False, True)
        """
        if other := self._isConstant(other):
            der_cp = None
            if other.der is None or self.der is None:
                warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
            else:
                der_cp = self.der >= other.der
            return self.val >= other.val, der_cp

    def __eq__(self, other):
        """
        Return element-wise (value and derivative vector) equality comparison of
        `self` and `other`.

        Parameters
        ----------
        self : Node
        other : int, float, Node

        Returns
        -------
        out : tuple

        Notes
        -----
        If either `other` or self has None der. We return None for the derivative comparison and 
        give user a warning indicating that they are attempting to compare two Nodes before their
        derivatives are computed

        Examples
        --------
        >>> Node(42) == 5
        RuntimeWarning: Attempting to compare two nodes with None derivatives
        warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
        (False, None)
        >>> Node.constant(42) == Node.constant(5)
        (False, True)
        >>> Node.constant(42) == Node.constant(42)
        (True, True)
        """
        if other := self._isConstant(other):
            der_cp = None
            if other.der is None or self.der is None:
                warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
            else:
                der_cp = self.der == other.der
            return self.val == other.val, der_cp

    def __ne__(self, other):
        """
        Return element-wise (value and derivative vector) inequality comparison of
        `self` and `other`.

        Parameters
        ----------
        self : Node
        other : int, float, Node

        Returns
        -------
        out : tuple

        Notes
        -----
        If either `other` or self has None der. We return None for the derivative comparison and 
        give user a warning indicating that they are attempting to compare two Nodes before their
        derivatives are computed

        Examples
        --------
        >>> Node(42) != 5
        RuntimeWarning: Attempting to compare two nodes with None derivatives
        warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
        (True, None)
        >>> Node.constant(42) != Node.constant(5)
        (True, True)
        >>> Node.constant(42) != Node.constant(42)
        (False, True)
        """
        if other := self._isConstant(other):
            der_cp = None
            if other.der is None or self.der is None:
                warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
            else:
                der_cp = self.der != other.der
            return self.val != other.val, der_cp

    def __repr__(self):
        """
        Return a string representation of the Node.

        Parameters
        ----------
        self : Node

        Returns
        -------
        out : str
        """
        return f"{self.__class__.__name__}({self.val})"