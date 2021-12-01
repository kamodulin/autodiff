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


def test_dual_from_non_1d_array():
    with pytest.raises(Exception):
        ad.Dual.from_array([[1, 2], [3, 4]])


@pytest.mark.parametrize("val", [[0.7], [-64]])
def test_dual_from_array_single_val(val):
    x = ad.Dual.from_array(val)
    assert _equal(x, val[0], 1)


def test_dual_compat_mismatch_dims_error():
    with pytest.raises(ArithmeticError):
        x = ad.Dual(1)
        y = ad.Dual(1, [1, 2])
        return x + y


def test_dual_compat_type_error():
    with pytest.raises(TypeError):
        x = ad.Dual(1)
        y = "autodiff"
        return x + y


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_add_constant(val1, val2):
    x = val1
    y = ad.Dual.constant(val2)
    assert _equal(x + y, val1 + val2, 0)

    x = ad.Dual.constant(val1)
    y = val2
    assert _equal(x + y, val1 + val2, 0)

    x = ad.Dual.constant(val1)
    y = ad.Dual.constant(val2)
    assert _equal(x + y, val1 + val2, 0)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [-3.4, 6])
@pytest.mark.parametrize("der2", [-1, 5])
def test_add_univariate(val1, der1, val2, der2):
    x = val1
    y = ad.Dual(val2, der2)
    assert _equal(x + y, val1 + val2, der2)

    x = ad.Dual(val1, der1)
    y = val2
    assert _equal(x + y, val1 + val2, der1)

    x = ad.Dual(val1, der1)
    y = ad.Dual(val2, der2)
    assert _equal(x + y, val1 + val2, der1 + der2)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [np.array([-3.4, 6]), np.array([-1, 24.2])])
@pytest.mark.parametrize("der2", [np.array([-4, 2]), np.array([-1.1, 32])])
def test_add_multivariate(val1, der1, val2, der2):
    x = val1
    y = ad.Dual(val2, der2)
    assert _equal(x + y, val1 + val2, der2)

    x = ad.Dual(val1, der1)
    y = val2
    assert _equal(x + y, val1 + val2, der1)

    x = ad.Dual(val1, der1)
    y = ad.Dual(val2, der2)
    assert _equal(x + y, val1 + val2, der1 + der2)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_mul_constant(val1, val2):
    x = val1
    y = ad.Dual.constant(val2)
    assert _equal(x * y, val1 * val2, val1 * 0 + val2 * 0)

    x = ad.Dual.constant(val1)
    y = val2
    assert _equal(x * y, val1 * val2, val1 * 0 + val2 * 0)

    x = ad.Dual.constant(val1)
    y = ad.Dual.constant(val2)
    assert _equal(x * y, val1 * val2, val1 * 0 + val2 * 0)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [-3.4, 6])
@pytest.mark.parametrize("der2", [-1, 5])
def test_mul_univariate(val1, der1, val2, der2):
    x = val1
    y = ad.Dual(val2, der2)
    assert _equal(x * y, val1 * val2, val1 * der2 + val2 * 0)

    x = ad.Dual(val1, der1)
    y = val2
    assert _equal(x * y, val1 * val2, val1 * 0 + val2 * der1)

    x = ad.Dual(val1, der1)
    y = ad.Dual(val2, der2)
    assert _equal(x * y, val1 * val2, val1 * der2 + val2 * der1)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [np.array([-3.4, 6]), np.array([-1, 24.2])])
@pytest.mark.parametrize("der2", [np.array([-4, 2]), np.array([-1.1, 32])])
def test_mul_multivariate(val1, der1, val2, der2):
    x = val1
    y = ad.Dual(val2, der2)
    assert _equal(x * y, val1 * val2, val1 * der2 + val2 * 0)

    x = ad.Dual(val1, der1)
    y = val2
    assert _equal(x * y, val1 * val2, val1 * 0 + val2 * der1)

    x = ad.Dual(val1, der1)
    y = ad.Dual(val2, der2)
    assert _equal(x * y, val1 * val2, val1 * der2 + val2 * der1)


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_pow_constants(val1, val2):
    x = val1
    y = ad.Dual.constant(val2)
    assert _equal(x**y, val1**val2, 0)

    x = ad.Dual.constant(val1)
    y = val2
    assert _equal(x**y, val1**val2, 0)

    x = ad.Dual.constant(val1)
    y = ad.Dual.constant(val2)
    assert _equal(x**y, val1**val2, 0)


@pytest.mark.parametrize("val1", [-0.7, -64])
@pytest.mark.parametrize("val2", [-2.1, 4.2])
def test_pow_constants_invalid(val1, val2):
    with pytest.raises(ValueError):
        x = val1
        y = ad.Dual.constant(val2)
        assert _equal(x**y, val1**val2, 0)

    with pytest.raises(ValueError):
        x = ad.Dual.constant(val1)
        y = val2
        assert _equal(x**y, val1**val2, 0)

    with pytest.raises(ValueError):
        x = ad.Dual.constant(val1)
        y = ad.Dual.constant(val2)
        assert _equal(x**y, val1**val2, 0)


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [-3.4, 6])
@pytest.mark.parametrize("der2", [-1, 5])
def test_pow_univariate(val1, der1, val2, der2):
    x = val1
    y = ad.Dual(val2, der2)
    assert _equal(x**y, val1**val2, val1**val2 * np.log(val1) * der2)

    x = ad.Dual(val1, der1)
    y = val2
    assert _equal(x**y, val1**val2, val2 * val1**(val2 - 1) * der1)

    x = ad.Dual(val1, der1)
    y = ad.Dual(val2, der2)
    int_der = der2 * np.log(val1) + val2 * (der1 / val1)
    assert _equal(x**y, val1**val2, val1**val2 * int_der)


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [np.array([-3.4, 6]), np.array([-1, 24.2])])
@pytest.mark.parametrize("der2", [np.array([-4, 2]), np.array([-1.1, 32])])
def test_pow_multivariate(val1, der1, val2, der2):
    x = val1
    y = ad.Dual(val2, der2)
    assert _equal(x**y, val1**val2, val1**val2 * np.log(val1) * der2)

    x = ad.Dual(val1, der1)
    y = val2
    assert _equal(x**y, val1**val2, val2 * val1**(val2 - 1) * der1)

    x = ad.Dual(val1, der1)
    y = ad.Dual(val2, der2)
    int_der = der2 * np.log(val1) + val2 * (der1 / val1)
    assert _equal(x**y, val1**val2, val1**val2 * int_der)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_sub_constant(val1, val2):
    x = val1
    y = ad.Dual.constant(val2)
    assert _equal(x - y, val1 - val2, 0)

    x = ad.Dual.constant(val1)
    y = val2
    assert _equal(x - y, val1 - val2, 0)

    x = ad.Dual.constant(val1)
    y = ad.Dual.constant(val2)
    assert _equal(x - y, val1 - val2, 0)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [-3.4, 6])
@pytest.mark.parametrize("der2", [-1, 5])
def test_sub_univariate(val1, der1, val2, der2):
    x = val1
    y = ad.Dual(val2, der2)
    assert _equal(x - y, val1 - val2, 0 - der2)

    x = ad.Dual(val1, der1)
    y = val2
    assert _equal(x - y, val1 - val2, der1 - 0)

    x = ad.Dual(val1, der1)
    y = ad.Dual(val2, der2)
    assert _equal(x - y, val1 - val2, der1 - der2)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [np.array([-3.4, 6]), np.array([-1, 24.2])])
@pytest.mark.parametrize("der2", [np.array([-4, 2]), np.array([-1.1, 32])])
def test_sub_multivariate(val1, der1, val2, der2):
    x = val1
    y = ad.Dual(val2, der2)
    assert _equal(x - y, val1 - val2, 0 - der2)

    x = ad.Dual(val1, der1)
    y = val2
    assert _equal(x - y, val1 - val2, der1 - 0)

    x = ad.Dual(val1, der1)
    y = ad.Dual(val2, der2)
    assert _equal(x - y, val1 - val2, der1 - der2)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_truediv_constant(val1, val2):
    x = val1
    y = ad.Dual.constant(val2)
    assert _equal(x / y, val1 / val2, 0)

    x = ad.Dual.constant(val1)
    y = val2
    assert _equal(x / y, val1 / val2, 0)

    x = ad.Dual.constant(val1)
    y = ad.Dual.constant(val2)
    assert _equal(x / y, val1 / val2, 0)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [-3.4, 6])
@pytest.mark.parametrize("der2", [-1, 5])
def test_truediv_univariate(val1, der1, val2, der2):
    x = val1
    y = ad.Dual(val2, der2)
    assert _equal(x / y, val1 / val2, (-val1 * der2) / (val2**2))

    x = ad.Dual(val1, der1)
    y = val2
    assert _equal(x / y, val1 / val2, (der1 * val2) / (val2**2))

    x = ad.Dual(val1, der1)
    y = ad.Dual(val2, der2)
    assert _equal(x / y, val1 / val2, (val2 * der1 - val1 * der2) / (val2**2))


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [np.array([-3.4, 6]), np.array([-1, 24.2])])
@pytest.mark.parametrize("der2", [np.array([-4, 2]), np.array([-1.1, 32])])
def test_truediv_multivariate(val1, der1, val2, der2):
    x = val1
    y = ad.Dual(val2, der2)
    assert _equal(x / y, val1 / val2, (-val1 * der2) / (val2**2))

    x = ad.Dual(val1, der1)
    y = val2
    assert _equal(x / y, val1 / val2, (der1 * val2) / (val2**2))

    x = ad.Dual(val1, der1)
    y = ad.Dual(val2, der2)
    assert _equal(x / y, val1 / val2, (val2 * der1 - val1 * der2) / (val2**2))


def test_neg_constants():
    x = ad.Dual.constant(2)
    y = ad.Dual.constant(-2)
    val = True
    der = True
    assert _compare((-x == y), val, der)


def test_neg_univariate():
    x, y = ad.Dual(-1, 11), ad.Dual(1, 11)
    val = True
    der = False
    assert _compare((-x == y), val, der)


def test_lt_multivariate():
    x, y = ad.Dual.from_array([6, 4])
    val = False
    der = [True, True]
    assert _compare((-x == y), val, der)


def test_lt_constants():
    x = ad.Dual.constant(2)
    y = ad.Dual.constant(3)
    val = True
    der = False
    assert _compare((x < y), val, der)


def test_lt_univariate():
    x, y = ad.Dual(1, 11), ad.Dual(2, 20)
    val = True
    der = True
    assert _compare((x < y), val, der)


def test_lt_multivariate():
    x, y = ad.Dual.from_array([6, -6])
    val = True
    der = [False, False]
    assert _compare((-x == y), val, der)


def test_gt_constants():
    x = ad.Dual.constant(1)
    y = ad.Dual.constant(2)
    val = False
    der = False
    assert _compare((x > y), val, der)


def test_gt_univariate():
    x, y = ad.Dual(1, 11), ad.Dual(2, -5)
    val = False
    der = True
    assert _compare((x > y), val, der)


def test_gt_multivariate():
    x, y = ad.Dual.from_array([8, 2])
    val = True
    der = [True, False]
    assert _compare((x > y), val, der)


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
