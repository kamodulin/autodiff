import numpy as np

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
                raise ValueError(f'Cannot set gradient to zero for type {self.__class__.__name__}')

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

    def _isConstant(self,other):
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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.val})"