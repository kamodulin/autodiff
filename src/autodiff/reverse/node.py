import numpy as np
import warnings

class Node:
    def __init__(self, val):
        self.val = val
        self.children = []
        self.der = None

    def grad(self):
        if self.children is not None and len(self.children) == 0:
            return 1.0
        if self.der is None:
            self.der = sum(w*node.grad() for w, node in self.children)
        return self.der

    # Experimental grad(): calls 
    # Params: variables with which to take the derivatives 
    # Returns: list of derivatives
    # Sample use case: f.grad(x,y,z)
    
    # def grad(self, *args):
    #     if len(args) == 0:
    #         return [1.0]
    #     grads = []
    #     for to in args:
    #         grads.append(self._gradrecur(to))
    #     return grads

    # Experimental feature
    # def _gradrecur(self, wrt):
    #     if wrt is self:
    #         return 1.0
    #     if wrt.der is None:
    #         wrt.der = sum(w*self._gradrecur(node) for w, node in wrt.children)
    #     return wrt.der

    @staticmethod
    def zero_grad(*args):
        for arg in args:
            try:
                arg.children = []
                arg.der = None
            except AttributeError:
                raise AttributeError(f'Cannot set gradient to zero for type {arg.__class__.__name__}')

    @staticmethod
    def constant(val):
        node = Node(val)
        node.children = None
        node.der = 0
        return node

    @staticmethod
    def from_array(X):
        if np.ndim(X) != 1:
            raise Exception(f"array must be 1-dimensional")
        if len(X) == 1:
            return Node(X[0])

        return iter(Node(x) for x in X)

    def _isConstant(self, other, operand = None):
        if isinstance(other, (int, float)):
            return Node.constant(other)
        elif isinstance(other, Node):
            return other
        raise TypeError(f"unsupported operand type(s) for {operand}: '{type(self).__name__}' and '{type(other).__name__}'")


    def _addChildren(self, new_weight, new_child):
        if self.children is not None:
            self.children.append((new_weight, new_child))

    def __add__(self,other):
        if other := self._isConstant(other):
            child = Node(self.val + other.val)
            self._addChildren(1.0,child)
            other._addChildren(1.0,child)
            return child

    def __radd__(self,other):
        return self + other

    def __mul__(self,other):
        if other := self._isConstant(other):
            child = Node(self.val*other.val)
            self._addChildren(other.val, child)
            other._addChildren(self.val, child)
            return child

    def __rmul__(self,other):
        return self * other

    def __sub__(self,other):
        if other := self._isConstant(other):
            child = Node(self.val - other.val)
            self._addChildren(1.0,child)
            other._addChildren(-1.0,child)
            return child

    def __rsub__(self,other):
        if other := self._isConstant(other):
            child = Node(other.val - self.val)
            self._addChildren(-1.0,child)
            other._addChildren(1.0,child)
            return child

    def __truediv__(self, other):
        if other := self._isConstant(other):
            child = Node(self.val/other.val)
            self._addChildren(1/other.val,child)
            other._addChildren(-self.val/(other.val**2),child)
            return child

    def __rtruediv__(self, other):
        if other := self._isConstant(other):
            child = Node(other.val/self.val)
            self._addChildren(-other.val/(self.val**2),child)
            other._addChildren(1/self.val,child)
            return child
    
    def __pow__(self, other):
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
        if other <= 0:
            raise ValueError(
                f"{other} cannot be raised to the power of {self.val}; log is undefined for x = {other}"
            )
        val = other**self.val
        child = Node(val)
        self._addChildren(val*np.log(other),child)
        return child

    def __neg__(self):
        if self.children == None:
            child = Node.constant(-self.val)
        else:
            child = Node(-self.val)
            self._addChildren(-1.0,child)
        return child

    def __lt__(self, other):
        if other := self._isConstant(other):
            der_cp = None
            if other.der is None or self.der is None:
                warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
            else:
                der_cp = self.der < other.der
            return self.val < other.val, der_cp

    def __gt__(self, other):
        if other := self._isConstant(other):
            der_cp = None
            if other.der is None or self.der is None:
                warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
            else:
                der_cp = self.der > other.der
            return self.val > other.val, der_cp

    def __le__(self, other):
        if other := self._isConstant(other):
            der_cp = None
            if other.der is None or self.der is None:
                warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
            else:
                der_cp = self.der <= other.der
            return self.val <= other.val, der_cp

    def __ge__(self, other):
        if other := self._isConstant(other):
            der_cp = None
            if other.der is None or self.der is None:
                warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
            else:
                der_cp = self.der >= other.der
            return self.val >= other.val, der_cp

    def __eq__(self, other):
        if other := self._isConstant(other):
            der_cp = None
            if other.der is None or self.der is None:
                warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
            else:
                der_cp = self.der == other.der
            return self.val == other.val, der_cp

    def __ne__(self, other):
        if other := self._isConstant(other):
            der_cp = None
            if other.der is None or self.der is None:
                warnings.warn('Attempting to compare two nodes with None derivatives',RuntimeWarning)
            else:
                der_cp = self.der != other.der
            return self.val != other.val, der_cp

    def __repr__(self):
        return f"{self.__class__.__name__}({self.val})"