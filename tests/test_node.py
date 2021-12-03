import numpy as np
import pytest

import autodiff.reverse as ad

from utils import _equal, _compare_node


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_add_constant(val1, val2):
    x = val1
    y = ad.Node.constant(val2)
    f = x + y
    der = 0
    eval_der = np.array([y.grad()])
    assert _equal(f, val1 + val2, der, eval_der)

    x = ad.Node.constant(val1)
    y = val2
    f = x + y
    der = 0
    eval_der = np.array([x.grad()])
    assert _equal(f, val1 + val2, der, eval_der)

    x = ad.Node.constant(val1)
    y = ad.Node.constant(val2)
    f = x + y
    der = 0 + 0
    eval_der = np.array([x.grad(), y.grad()])
    assert _equal(f, val1 + val2, der, eval_der)


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_add_variable(val1, val2):
    x = val1
    y = ad.Node(val2)
    f = x + y
    der = 1
    eval_der = np.array([y.grad()])
    assert _equal(f, val1 + val2, der, eval_der)

    x = ad.Node(val1)
    y = val2
    f = x + y
    der = 1
    eval_der = np.array([x.grad()])
    assert _equal(f, val1 + val2, der, eval_der)

    x = ad.Node(val1)
    y = ad.Node(val2)
    f = x + y
    der = np.array([1, 1])
    eval_der = np.array([x.grad(), y.grad()])
    assert _equal(f, val1 + val2, der, eval_der)


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_pow_constant(val1, val2):
    x = val1
    y = ad.Node.constant(val2)
    f = x**y
    der = 0
    eval_der = np.array([y.grad()])
    assert _equal(f, val1**val2, der, eval_der)

    x = ad.Node.constant(val1)
    y = val2
    f = x**y
    der = 0
    eval_der = np.array([x.grad()])
    assert _equal(f, val1**val2, der, eval_der)

    x = ad.Node.constant(val1)
    y = ad.Node.constant(val2)
    f = x**y
    der = np.array([0, 0])
    eval_der = np.array([x.grad(), y.grad()])
    assert _equal(f, val1**val2, der, eval_der)


@pytest.mark.parametrize("val1", [-0.7, -64])
@pytest.mark.parametrize("val2", [-2.1, 4.2])
def test_pow_invalid(val1, val2):
    with pytest.raises(ValueError):
        x = val1
        y = ad.Node.constant(val2)
        _ = x**y

        x = ad.Node.constant(val1)
        y = val2
        _ = x**y

        x = ad.Node.constant(val1)
        y = ad.Node.constant(val2)
        _ = x**y

    with pytest.raises(ZeroDivisionError):
        x = ad.Node.constant(0)
        y = -2.1
        _ = x**y


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_pow_variable(val1, val2):
    x = val1
    y = ad.Node(val2)
    f = x**y
    der = val1**(val2) * np.log(val1)
    eval_der = np.array([y.grad()])
    assert _equal(f, val1**val2, der, eval_der)

    x = ad.Node(val1)
    y = val2
    f = x**y
    der = val1**(val2) * val2 / val1
    eval_der = np.array([x.grad()])
    assert _equal(f, val1**val2, der, eval_der)

    x = ad.Node(val1)
    y = ad.Node(val2)
    f = x**y
    x_grad = val1**(val2) * val2 / val1
    y_grad = val1**(val2) * np.log(val1)
    der = np.array([x_grad, y_grad])
    eval_der = np.array([x.grad(), y.grad()])
    assert _equal(f, val1**val2, der, eval_der)