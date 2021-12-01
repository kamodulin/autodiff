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

# x,y,z = Node.from_array([1,2,3])
# f = 3/(x+2)*(9/z) + 3/(x/ (1*x - z/(x-2)*(3+y) + x/1 + x/(2*(1-z)*(x+1)) - y/(6*(2-x)) + 7 + (3-y)))/x/y/3

# print(x.grad())
# print(y.grad())
# print(z.grad())