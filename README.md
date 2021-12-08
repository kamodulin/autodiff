# cs107-FinalProject - Group #32
## Members: Xuliang Guo, Kamran Ahmed, Van Anh Le, Hanwen Cui

[![codecov](https://codecov.io/gh/cs107-XKVH/cs107-FinalProject/branch/main/graph/badge.svg?token=SAQEVYPUXC)](https://codecov.io/gh/cs107-XKVH/cs107-FinalProject)

## Broader Impacts and Inclusivity

### Broader Impacts
Virtually all machine learning and AI algorithms can be attributed to solving optimization problems during the training process. While automatic differentiation does not direct broader impacts, its extensive use as an intermediate step in these algorithms forces us to consider the broader impact of our package. First of all, our package will be contributing to biases against African-American and other underrepresented minorities that current ML models used in the criminal justice system or hiring processes are already imposing. Second, any errors in our calculations could lead to misspecified models and erroneous predictions with significant impacts to downstream users. These impacts are especially grave in safety-critical settings such as healthcare, where a model that utilizes a faulty AD library could misdiagnose a patient or suggest sub-optimal treatments. 

### Inclusivity
While our codebase is technically available and open for anyone to contribute through our GitHub repository, there are technical barriers that might prevent certain groups from participating in this process. Any contributors would need to have working knowledge of git version control and principles of software development. This precludes people from rural communities, communities of color, or poor urban communities, who are less likely to receive formal and rigorous training in computer science. Even at the college level, CS curricula are not homogenous and concepts such as git version control might not be taught at every school. Furthermore, users from other disciplines who rely on optimization and AD might be discouraged by the initial fixed cost of learning a complicated system such as git.

Any developer who wants to contribute to our codebase can make a new branch and create a pull request. Pull requests will then be reviewed by one or many members of our team, depending on the extent of contribution. In order to make this process more inclusive, we could include a step-by-step guide on our repository that provides explicit direction on how to work with git and the expected best-practices that we hope they would follow.

## How to install
We recommend creating a virtual environment rather than installing in the base environment:
```
python3 -m venv autodiff-env
source autodiff/bin/activate
```

Our package can be installed from Github or PyPI. We also include source distribution files and wheels under Releases.

You can install ac207-autodiff via `pip` with:
```
pip install ac207-autodiff
```

## Basic usage
Detailed descriptions about classes, methods, and operations can be found in our [API reference](https://cs107-xkvh.github.io/).

Our automatic differentiation package’s default behavior uses forward mode. You can import this as follows:
```python
import autodiff as ad
```

If you would like to use reverse mode, please explicitly import it as:
```python
import autodiff.reverse as ad
```

### Forward mode
The properties of a dual number lend itself nicely to a straightforward implementation of forward mode automatic differentiation. Briefly, we use dual numbers as our core data structure (`ad.Dual`). The value and derivative can be stored as the real and “dual” part of the dual number, respectively.

We provide support for:
- Most arithmetic and comparison operations
- Elementary operations such as trigonometric functions, square root, logarithmic, logistic, and exponential functions, among others.

**Univariate functions**
```python
>>> import autodiff as ad
>>> x = ad.Dual(2)
>>> f = 7 * (x ** 3) + 3 * x
>>> print(f"Function value: {f.val}, derivative: {f.der}")
Function value: 62, derivative: [87.]
```

**Multivariate functions**
```python
>>> import autodiff as ad
>>> x, y = ad.Dual.from_array([2, 4]) # helper static method
>>> f = 7 * (x ** 3) + 3 * y
>>> print(f"Function value: {f.val}, derivative: {f.der}")
Function value: 68, derivative: [84. 3.]
```

**Vector functions**
```python
>>> import autodiff as ad
>>> def f(x, y, z): # Vector function mapping 3 inputs to 2 outputs.
...    f1 = 7 * (x ** 3) + 3 * y
...    f2 = y / x + z ** 2
...    return (f1, f2)
...
>>> x, y, z = ad.Dual.from_array([2, 4, 6])
>>> f1, f2 = f(x, y, z)
>>> print(f"f1 value: {f1.val}, derivative: {f1.der}")
f1 value: 68, derivative: [84.  3.  0.]
>>> print(f"f2 value: {f2.val}, derivative: {f2.der}")
f2 value: 38.0, derivative: [-1.   0.5 12. ]
```

**Elementary operations**
```python
>>> import autodiff as ad
>>> x, y = ad.Dual.from_array([2, 4])
>>> f = ad.exp(x) + y
>>> print(f"Function value: {f.val:.4f}, " \
... 		 "derivative: [{f.der[0]:.4f} {f.der[1]:.4f}]")
Function value: 11.3891, derivative: [7.3891 1.0]
```

### Reverse mode
Note that these are contained within the `autodiff.reverse` module. 

Explicitly import it as:
```python
>>> import autodiff.reverse as ad
```

`ad.Node` is the primary data structure for reverse mode automatic differentiation. The process of evaluating derivatives in reverse mode consists of two passes, forward pass and reverse pass. During the forward pass, we calculate the primal values and the local gradient of child nodes with respect of each parent node in the computational graph. In the reverse pass, we recursively calculate the gradients.

Reverse mode only evalates the function at the specified values. To calculate the gradient with respect to each input, you have to explicitly call `Node.grad()`. Examples can be found below.

**Univariate function**

The derivative of the function is not stored within the function object, but rather is computed on the fly when `x.grad()` is called.
```python
>>> import autodiff.reverse as ad
>>> x = ad.Node(2)
>>> f = 7 * (x ** 3) + 3 * x
>>> grad = x.grad() # compute gradient
>>> print(f"Function value: {f.val}, derivative w.r.t x = {grad}")
Function value: 62, derivative w.r.t x = 87.0
```

Note that to reuse the `x` variable again, without accumulating gradients you must call `ad.Node.zero_grad(x)`. A more detailed example can be found below when using vector functions.

**Multivariate functions**
```python
>>> import autodiff.reverse as ad
>>> x = ad.Node(2)
>>> y = ad.Node(4)
>>> f = 7 * (x ** 3) + 3 * y
>>> grad = [x.grad(), y.grad()] # explicitly compute all gradients w.r.t. x and y
>>> print(f"Function value: {f.val}, derivative: {grad}")
Function value: 68, derivative: [84.0, 3.0]
```

**Vector functions**
```python
>>> import autodiff.reverse as ad
>>> x, y, z = ad.Node.from_array([2, 4, 6])
>>> def f(x, y, z): # Vector function mapping 3 inputs to 2 outputs
...     f1 = 7 * (x ** 3) + 3 * y
...     f1_grad = [x.grad(), y.grad(), z.grad()] # compute gradient w.r.t. all inputs, before computing f2
...     ad.Node.zero_grad(x, y, z)  # must be called before computing f2, otherwise gradients will accumulate
...     f2 = y / x + z ** 2
...     f2_grad = [x.grad(), y.grad(), z.grad()]
...     return f1, f1_grad, f2, f2_grad
>>> f1, f1_grad, f2, f2_grad = f(x, y, z)
>>> print(f"First function value: {f1.val}, derivative: {f1_grad}")
First function value: 68, derivative: [84.0, 3.0, 1.0]
>>> print(f"Second function value: {f2.val}, derivative: {f2_grad}")
Second function value: 38.0, derivative: [-1.0, 0.5, 12.0]
```

**Elementary operations**

We allow users to import overloaded elementary functions (sine, cosine, tangent, exponential, log, sqrt) to perform operations on Nodes.
```python
>>> import autodiff.reverse as ad
>>> x, y = ad.Node.from_array([2, 4])
>>> f = ad.exp(x) + y
>>> grad = [x.grad(), y.grad()]
>>> print(f"Function value: {f.val:.4f}, derivative: [{grad[0]:.4f} {grad[1]:.4}]")
Function value: 11.3891, derivative: [7.3891 1.0]
```