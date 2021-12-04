import numpy as np
import pytest

import autodiff.reverse as ad

from utils import _equal, _compare_node


@pytest.mark.parametrize("val", [1, -6.2])
def test_node_constant(val):
    x = ad.Node.constant(val)
    assert _equal(x, val, 0, np.array([x.grad()]))


@pytest.mark.parametrize("val", [0.7, -64])
def test_node_variable(val):
    x = ad.Node(val)
    assert _equal(x, val, 1.0, np.array([x.grad()]))


@pytest.mark.parametrize("vals", [np.array([-3.4, 6]), np.array([-1, 6])])
def test_node_from_array(vals):
    xs = list(ad.Node.from_array(vals))

    for x, val in zip(xs, vals):
        assert _equal(x, val, 1.0, np.array([x.grad()]))


def test_node_from_non_1d_array():
    with pytest.raises(Exception):
        ad.Node.from_array([[1, 2], [3, 4]])


@pytest.mark.parametrize("val", [[0.7], [-64]])
def test_node_from_array_single_val(val):
    x = ad.Node.from_array(val)
    assert _equal(x, val[0], 1, np.array([x.grad()]))


def test_node_compat_type_error():
    with pytest.raises(TypeError):
        x = ad.Node(1)
        y = "autodiff"
        return x + y


@pytest.mark.parametrize("vals", [np.array([-3.4, 6]), np.array([-1, 6])])
@pytest.mark.parametrize("ders", [[-0.3, 6], [3, -6.3]])
def test_zero_grad(vals, ders):
    nodes = list(ad.Node.from_array(vals))
    for i, n in enumerate(nodes):
        n.der = ders[i]
    for i, n in enumerate(nodes):
        ad.Node.zero_grad(n)
        assert _equal(n, vals[i], 1.0, np.array([n.grad()]))


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

    with pytest.raises(ValueError):
        x = ad.Node.constant(val1)
        y = val2
        _ = x**y
    
    with pytest.raises(ValueError):
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


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_mul_constant(val1, val2):
    x = ad.Node.constant(val1)
    y = ad.Node.constant(val2)
    assert _equal(x * y, val1 * val2, 0, np.array([x.grad(), y.grad()]))


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_mul_variable(val1, val2):
    x = ad.Node(val1)
    y = ad.Node(val2)
    assert _equal(x * y, val1 * val2, np.array([val2, val1]),
                  np.array([x.grad(), y.grad()]))


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_sub_constant(val1, val2):
    x = ad.Node.constant(val1)
    y = ad.Node.constant(val2)
    assert _equal(x - y, val1 - val2, 0, np.array([x.grad(), y.grad()]))


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_sub_variable(val1, val2):
    x = ad.Node(val1)
    y = ad.Node(val2)
    assert _equal(x - y, val1 - val2, np.array([1.0, -1.0]),
                  np.array([x.grad(), y.grad()]))


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_rsub_constant(val1, val2):
    x = val1
    y = ad.Node(val2)
    assert _equal(x - y, val1 - val2, np.array([-1.0]), np.array([y.grad()]))


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_truediv_variable(val1, val2):
    x = ad.Node(val1)
    y = ad.Node(val2)
    assert _equal(x / y, val1 / val2, np.array([1 / val2, -val1 / (val2**2)]),
                  np.array([x.grad(), y.grad()]))


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_rtruediv_variable(val1, val2):
    x = val1
    y = ad.Node(val2)
    assert _equal(x / y, val1 / val2, np.array([-val1 / (val2**2)]),
                  np.array([y.grad()]))


def test_neg_constants():
    x = ad.Node.constant(2)
    y = ad.Node.constant(-2)
    f = (-x == y)
    val = True
    der = True
    assert _compare_node(f[0], val, der, f[1])


def test_neg_variable():
    x = ad.Node(2)
    out = -x
    assert _equal(out, -2, -1, np.array([x.grad()]))

    y = ad.Node.constant(2)
    out = -y
    assert _equal(out, -2, 0, np.array([y.grad()]))
