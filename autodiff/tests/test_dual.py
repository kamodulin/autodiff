import numpy as np
import pytest

import autodiff as ad

from utils import _equal, _compare


@pytest.mark.parametrize("val", [1, -6.2])
def test_dual_constant(val):
    x = ad.Dual.constant(val)
    assert _equal(x, val, 0)


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [-2, 4.2])
def test_dual_univariate(val, der):
    x = ad.Dual(val, der)
    assert _equal(x, val, der)


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([-1, 6])])
def test_dual_multivariate(val, der):
    x = ad.Dual(val, der)
    assert _equal(x, val, der)


@pytest.mark.parametrize("vals", [np.array([-3.4, 6]), np.array([-1, 6])])
def test_dual_from_array(vals):
    xs = list(ad.Dual.from_array(vals))

    for x, val, der in zip(xs, vals, np.identity(len(vals))):
        assert _equal(x, val, der)


# Add tests for arithmetic operations
def test_pow_constants():
    x = ad.Dual.constant(2)
    y = ad.Dual.constant(3)
    assert _equal(x**y, 8, 0)

def test_pow_dual_constant():
    x = ad.Dual(2, 2)
    assert _equal(x**3, 8, 24)
    
def test_pow_constant_dual():
    x = ad.Dual(4, 2)
    assert _equal(3**x, 81, 177.97519076)
    
def test_pow_dual_dual():
    x = ad.Dual(4, 2)
    assert _equal(x**x, 256, 1221.78271289)
    
def test_pow_multi():
    x, y = ad.Dual.from_array([2, 2])
    assert _equal(x**y, 4, 2.77258872)
    
# Add more comparison tests (e.g. <, >)


def test_le_constants():
    x = ad.Dual.constant(1)
    y = ad.Dual.constant(2)
    val = True
    der = True
    assert _compare((x <= y), val, der)


def test_le_univariate():
    x, y = ad.Dual(1, 11), ad.Dual(2, 2)
    val = True
    der = False
    assert _compare((x <= y), val, der)


def test_le_multivariate():
    x, y = ad.Dual.from_array([6, 4])
    val = False
    der = [False, True]
    assert _compare((x <= y), val, der)


def test_ge_constants():
    x = ad.Dual.constant(2.6)
    y = ad.Dual.constant(1.2)
    val = True
    der = True
    assert _compare((x >= y), val, der)


def test_ge_univariate():
    x, y = ad.Dual(1, 11), ad.Dual(2, -8)
    val = False
    der = True
    assert _compare((x >= y), val, der)


def test_ge_multivariate():
    x, y = ad.Dual.from_array([6, 2])
    val = True
    der = [True, False]
    assert _compare((x >= y), val, der)


def test_eq_constants():
    x = ad.Dual.constant(-6.4)
    y = ad.Dual.constant(3)
    val = False
    der = True
    assert _compare((x == y), val, der)


def test_eq_univariate():
    x, y = ad.Dual(5, -9), ad.Dual(20, -9)
    val = False
    der = True
    assert _compare((x == y), val, der)


def test_eq_multivariate():
    x, y = ad.Dual.from_array([2.8, 2.8])
    val = True
    der = [False, False]
    assert _compare((x == y), val, der)


def test_ne_constants():
    x = ad.Dual.constant(8)
    y = ad.Dual.constant(8)
    val = False
    der = False
    assert _compare((x != y), val, der)


def test_ne_univariate():
    x, y = ad.Dual(7, -6), ad.Dual(7, -6)
    val = False
    der = False
    assert _compare((x != y), val, der)


def test_ne_multivariate():
    x, y = ad.Dual.from_array([1, 2])
    val = True
    der = [True, True]
    assert _compare((x != y), val, der)
