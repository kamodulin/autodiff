import numpy as np
import pytest

import autodiff.reverse as ad

from utils import _equal


@pytest.mark.parametrize("val", [1, -6.2])
def test_sin_number(val):
    x = np.sin(val)
    out = ad.sin(val)
    assert pytest.approx(x, out)


@pytest.mark.parametrize("val", [1, -6.2])
def test_sin_constant(val):
    x = ad.Node.constant(val)
    out = ad.sin(x)
    der = np.cos(val) * 0
    eval_der = np.array([x.grad()])
    assert _equal(out, np.sin(val), der, eval_der)


@pytest.mark.parametrize("val", [0.7, -64])
def test_sin_variable(val):
    x = ad.Node(val)
    out = ad.sin(x)
    der = np.cos(val)
    eval_der = np.array([x.grad()])
    assert _equal(out, np.sin(val), der, eval_der)


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [0.7, -64])
@pytest.mark.parametrize("child_val", [0.7, -64])
def test_sin_multichildren(val, der, child_val):
    x = ad.Node(val)
    child = ad.Node(child_val)
    x._addChildren(der, child)
    out = ad.sin(x)
    der = np.cos(val) + der
    eval_der = np.array([x.grad()])
    assert _equal(out, np.sin(val), der, eval_der)


@pytest.mark.parametrize("val", [1, -6.2])
def test_cos_number(val):
    x = np.cos(val)
    out = ad.cos(val)
    assert pytest.approx(x, out)


@pytest.mark.parametrize("val", [1, -6.2])
def test_cos_constant(val):
    x = ad.Node.constant(val)
    out = ad.cos(x)
    der = -np.sin(val) * 0
    eval_der = np.array([x.grad()])
    assert _equal(out, np.cos(val), der, eval_der)


@pytest.mark.parametrize("val", [0.7, -64])
def test_cos_variable(val):
    x = ad.Node(val)
    out = ad.cos(x)
    der = -np.sin(val)
    eval_der = np.array([x.grad()])
    assert _equal(out, np.cos(val), der, eval_der)


@pytest.mark.parametrize("val", [1, -6.2])
def test_tan_number(val):
    x = np.tan(val)
    out = ad.tan(val)
    assert pytest.approx(x, out)


@pytest.mark.parametrize("val", [1, -6.2])
def test_tan_constant(val):
    x = ad.Node.constant(val)
    out = ad.tan(x)
    der = 0 / (np.cos(val)**2)
    eval_der = np.array([x.grad()])
    assert _equal(out, np.tan(val), der, eval_der)


@pytest.mark.parametrize("val", [0.7, -64])
def test_tan_variable(val):
    x = ad.Node(val)
    out = ad.tan(x)
    der = 1 / (np.cos(val)**2)
    eval_der = np.array([x.grad()])
    assert _equal(out, np.tan(val), der, eval_der)


def test_tan_derivative_undefined():
    x = ad.Node(np.pi / 2)
    with pytest.raises(ValueError):
        ad.tan(x)


@pytest.mark.parametrize("val", [0.7, -64])
def test_sinh_number(val):
    x = np.sinh(val)
    out = ad.sinh(val)
    assert pytest.approx(x, out)


@pytest.mark.parametrize("val", [1, -6.2])
def test_sinh_constant(val):
    x = ad.Node.constant(val)
    out = ad.sinh(x)
    der = 0
    eval_der = np.array([x.grad()])
    assert _equal(out, np.sinh(val), der, eval_der)


@pytest.mark.parametrize("val", [0.7, -64])
def test_sinh_variable(val):
    x = ad.Node(val)
    out = ad.sinh(x)
    der = np.cosh(val)
    eval_der = np.array([x.grad()])
    assert _equal(out, np.sinh(val), der, eval_der)


@pytest.mark.parametrize("val", [0.7, -64])
def test_cosh_number(val):
    x = np.cosh(val)
    out = ad.cosh(val)
    assert pytest.approx(x, out)


@pytest.mark.parametrize("val", [1, -6.2])
def test_cosh_constant(val):
    x = ad.Node.constant(val)
    out = ad.cosh(x)
    der = 0
    eval_der = np.array([x.grad()])
    assert _equal(out, np.cosh(val), der, eval_der)


@pytest.mark.parametrize("val", [0.7, -64])
def test_cosh_variable(val):
    x = ad.Node(val)
    out = ad.cosh(x)
    der = np.sinh(val)
    eval_der = np.array([x.grad()])
    assert _equal(out, np.cosh(val), der, eval_der)


@pytest.mark.parametrize("val", [0.7, -64])
def test_tanh_number(val):
    x = np.tanh(val)
    out = ad.tanh(val)
    assert pytest.approx(x, out)


@pytest.mark.parametrize("val", [1, -6.2])
def test_tanh_constant(val):
    x = ad.Node.constant(val)
    out = ad.tanh(x)
    der = 0
    eval_der = np.array([x.grad()])
    assert _equal(out, np.tanh(val), der, eval_der)


@pytest.mark.parametrize("val", [0.7, -64])
def test_tanh_variable(val):
    x = ad.Node(val)
    out = ad.tanh(x)
    der = 1 - (np.tanh(val))**2
    eval_der = np.array([x.grad()])
    assert _equal(out, np.tanh(val), der, eval_der)


@pytest.mark.parametrize("val", [0.5, 0.1, -0.1, -0.99, 0])
def test_arcsin_number(val):
    out = ad.arcsin(val)
    assert pytest.approx(out, np.arcsin(val))


@pytest.mark.parametrize("val", [-0.5])
def test_arcsin_constant(val):
    x = ad.Node.constant(val)
    out = ad.arcsin(x)
    der = 1 / np.sqrt(1 - val**2) * 0
    eval_der = np.array([x.grad()])
    assert _equal(out, np.arcsin(val), der, eval_der)


@pytest.mark.parametrize("val", [-0.5])
def test_arcsin_variable(val):
    x = ad.Node(val)
    out = ad.arcsin(x)
    der = (1 / np.sqrt(1 - val**2))
    eval_der = np.array([x.grad()])
    assert _equal(out, np.arcsin(val), der, eval_der)


@pytest.mark.parametrize("val", [-2, 1.2])
def test_arcsin_undefined(val):
    x = ad.Node(val)
    with pytest.raises(ValueError):
        ad.arcsin(x)


@pytest.mark.parametrize("val", [0.5, 0.1, -0.1, -0.99, 0])
def test_arccos_number(val):
    out = ad.arccos(val)
    assert pytest.approx(out, np.arccos(val))


@pytest.mark.parametrize("val", [-0.5])
def test_arccos_constant(val):
    x = ad.Node.constant(val)
    out = ad.arccos(x)
    der = (-1 / np.sqrt(1 - val**2)) * 0
    eval_der = np.array([x.grad()])
    assert _equal(out, np.arccos(val), der, eval_der)


@pytest.mark.parametrize("val", [-0.5])
def test_arccos_variable(val):
    x = ad.Node(val)
    out = ad.arccos(x)
    der = -1 / np.sqrt(1 - val**2)
    eval_der = np.array([x.grad()])
    assert _equal(out, np.arccos(val), der, eval_der)


@pytest.mark.parametrize("val", [-2, 1.2])
def test_arccos_undefined(val):
    x = ad.Node(val)
    with pytest.raises(ValueError):
        ad.arccos(x)


@pytest.mark.parametrize("val", [0.7, 64, -0.3, -10, 11.4])
def test_arctan_number(val):
    out = ad.arctan(val)
    assert pytest.approx(out, np.arctan(val))


@pytest.mark.parametrize("val", [-0.5])
def test_arctan_constant(val):
    x = ad.Node.constant(val)
    out = ad.arctan(x)
    der = (1 / (1 + val**2)) * 0
    eval_der = np.array([x.grad()])
    assert _equal(out, np.arctan(val), der, eval_der)


@pytest.mark.parametrize("val", [-0.5])
def test_acrtan_variable(val):
    x = ad.Node(val)
    out = ad.arctan(x)
    der = 1 / (1 + val**2)
    eval_der = np.array([x.grad()])
    assert _equal(out, np.arctan(val), der, eval_der)


@pytest.mark.parametrize("val", [1, -6.2])
def test_exp_number(val):
    out = ad.exp(val)
    assert pytest.approx(out, np.exp(val))


@pytest.mark.parametrize("val", [1, -6.2])
def test_exp_constant(val):
    x = ad.Node.constant(val)
    out = ad.exp(x)
    der = 0
    eval_der = np.array([x.grad()])
    assert _equal(out, np.exp(val), der, eval_der)


@pytest.mark.parametrize("val", [1, -6.2])
def test_exp_variable(val):
    x = ad.Node(val)
    out = ad.exp(x)
    der = np.exp(val)
    eval_der = np.array([x.grad()])
    assert _equal(out, np.exp(val), der, eval_der)


@pytest.mark.parametrize("val", [1, 2])
def test_log_number(val):
    x = np.log(val)
    out = ad.log(val)
    assert pytest.approx(x, out)


@pytest.mark.parametrize("val", [-1, 0])
def test_log_number_undefined(val):
    with pytest.raises(ValueError):
        ad.log(val)


@pytest.mark.parametrize("val", [1, 2])
def test_log_constant(val):
    x = ad.Node.constant(val)
    out = ad.log(x)
    der = 0
    eval_der = np.array([x.grad()])
    assert _equal(out, np.log(val), der, eval_der)


@pytest.mark.parametrize("val", [0.7, 64])
def test_log_variable(val):
    x = ad.Node(val)
    out = ad.log(x)
    der = 1 / val
    eval_der = np.array([x.grad()])
    assert _equal(out, np.log(val), der, eval_der)


@pytest.mark.parametrize("val", [0, -2.4, -11])
def test_log_undefined(val):
    x = ad.Node(val)
    with pytest.raises(ValueError):
        ad.log(x)


@pytest.mark.parametrize("val", [0, -2.4, -11])
@pytest.mark.parametrize("base", [0, -1, -5.7])
def test_log_invalid_base(val, base):
    x = ad.Node(val)
    with pytest.raises(ValueError):
        ad.log(x, base)


@pytest.mark.parametrize("val", [1, 2])
@pytest.mark.parametrize("base", [2, 10, 6.2])
def test_log_any_base_number(val, base):
    x = np.log(val) / np.log(base)
    out = ad.log(val, base)
    assert pytest.approx(x, out)


@pytest.mark.parametrize("val", [1, 2])
@pytest.mark.parametrize("base", [2, 10, 6.2])
def test_log_any_base_constant(val, base):
    x = ad.Node.constant(val)
    out = ad.log(x, base)
    der = 0
    eval_der = np.array([x.grad()])
    assert _equal(out, np.log(val) / np.log(base), der, eval_der)


@pytest.mark.parametrize("val", [1, 2])
@pytest.mark.parametrize("base", [2, 10, 6.2])
def test_log_any_base_variable(val, base):
    x = ad.Node(val)
    out = ad.log(x, base)
    der = 1 / (val * np.log(base))
    eval_der = np.array([x.grad()])
    assert _equal(out, np.log(val) / np.log(base), der, eval_der)


@pytest.mark.parametrize("val", [1, 6.2])
def test_sqrt_number(val):
    x = np.sqrt(val)
    out = ad.sqrt(val)
    assert pytest.approx(x, out)


@pytest.mark.parametrize("val", [1, 6.2])
def test_sqrt_constant(val):
    x = ad.Node.constant(val)
    out = ad.sqrt(x)
    der = 0
    eval_der = np.array([x.grad()])
    assert _equal(out, np.sqrt(val), der, eval_der)


@pytest.mark.parametrize("val", [0.7, 64])
def test_sqrt_variable(val):
    x = ad.Node(val)
    out = ad.sqrt(x)
    der = 0.5 / np.sqrt(val)
    eval_der = np.array([x.grad()])
    assert _equal(out, np.sqrt(val), der, eval_der)


@pytest.mark.parametrize("val", [-2.4, -11])
def test_sqrt_undefined(val):
    x = ad.Node(val)
    with pytest.raises(ValueError):
        ad.sqrt(x)


@pytest.mark.parametrize("val", [0.7, 64, -0.5, 10, -10])
def test_logistic_number(val):
    g = lambda z: 1 / (1 + np.exp(-z))
    out = ad.logistic(val)
    assert pytest.approx(out, g(val))


@pytest.mark.parametrize("val", [1, -6.2])
def test_logistic_constant(val):
    g = lambda z: 1 / (1 + np.exp(-z))
    x = ad.Node.constant(val)
    out = ad.logistic(x)
    der = 0
    eval_der = np.array([x.grad()])
    assert _equal(out, g(val), der, eval_der)


@pytest.mark.parametrize("val", [0.7, -64])
def test_logistic_variable(val):
    g = lambda z: 1 / (1 + np.exp(-z))
    x = ad.Node(val)
    out = ad.logistic(x)

    out_val = g(val)
    out_der = g(val) * (1 - g(val))
    eval_der = np.array([x.grad()])
    assert _equal(out, out_val, out_der, eval_der)


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [0.7, -64])
@pytest.mark.parametrize("child_val", [0.7, -64])
def test_logistic_multichildren(val, der, child_val):
    g = lambda z: 1 / (1 + np.exp(-z))
    x = ad.Node(val)
    child = ad.Node(child_val)
    x._addChildren(der, child)
    out = ad.logistic(x)

    out_val = g(val)
    out_der = g(val) * (1 - g(val)) + der
    eval_der = np.array([x.grad()])
    assert _equal(out, out_val, out_der, eval_der)
