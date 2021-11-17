import numpy as np
import pytest

import autodiff as ad

#from utils import _equal, _compare

def _equal(x, val, der):
    return x.val == val and np.all(x.der == der)


def _compare(comparison, val, der):
    x = ad.Dual(*comparison)
    return _equal(x, val, der)



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
    assert _equal(3**x, 81, (81*(2*np.log(3)+4*(0/3))))
    
def test_pow_dual_dual():
    x = ad.Dual(4, 2)
    assert _equal(x**x, 256, 256*(2*np.log(4)+4*(2/4)))
    
def test_pow_multi():
    x, y = ad.Dual.from_array([2, 2])
    val = x.val ** y.val
    der = val * (y.der * np.log(x.val) + y.val * x.der / x.val)
    assert _equal(x**y, val, der)
    
def test_sub_constants():
    x = ad.Dual.constant(1)    # two constants
    y = ad.Dual.constant(2)
    assert _equal(x-y,-1,0)

def test_sub_univariate():
    x, y = ad.Dual(1, 11), ad.Dual(2, 2)  # two univariate
    assert _equal(x-y,-1,9)

    a = 2 # one number and one univariate
    assert _equal(x-a,-1,11)

def test_sub_multivariate():
    x, y = ad.Dual.from_array([6, 4]) # two multivariate
    assert _equal(x-y,2,[1.0,-1.0])

    z = 2
    assert _equal(x-z,4,[1.0,0.0])    # one multivariate and one number

def test_rsub_constants():
    x = ad.Dual.constant(1)    # number subtract one constants
    assert _equal(0-x,-1,0)

def test_rsub_multivariate():
    x, y = ad.Dual.from_array([6, 4]) # number subtract multivariate
    assert _equal(0-x,-6,[-1.0,0.0])

    z = 2
    assert _equal(0-y,-4,[0.0,-1.0])

def test_truediv_constants():
    x = ad.Dual.constant(1)    # two constants
    y = ad.Dual.constant(2)
    assert _equal(x/y,0.5,0)

def test_truediv_univariate():
    x, y = ad.Dual(1, 11), ad.Dual(2, 2)  # two univariate
    assert _equal(x/y,0.5,5.0)

    a = 2 # one number and one univariate
    assert _equal(x/a,0.5,11/2)

def test_truediv_multivariate():
    x, y = ad.Dual.from_array([6, 4]) # two multivariate
    assert _equal(x/y,6/4,[1/4,-3/8])

    z = 2
    assert _equal(x/z,3,[1/2,0.0])    # one multivariate and one number

# Add more comparison tests (e.g. <, >)
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


#################################



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
