import numpy as np
import pytest

import autodiff as ad

from utils import _equal


@pytest.mark.parametrize("val", [1, -6.2])
def test_sin_constant(val):
    x = ad.Dual.constant(val)
    out = ad.sin(x)
    assert _equal(out, np.sin(val), np.cos(val) * 0)


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [-2, 4.2])
def test_sin_univariate(val, der):
    x = ad.Dual(val, der)
    out = ad.sin(x)
    assert _equal(out, np.sin(val), np.cos(val) * der)


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([-1, 24.2])])
def test_sin_multivariate(val, der):
    x = ad.Dual(val, der)
    out = ad.sin(x)
    assert _equal(out, np.sin(val), np.cos(val) * der)


@pytest.mark.parametrize("val", [1, -6.2])
def test_cos_constant(val):
    x = ad.Dual.constant(val)
    out = ad.cos(x)
    assert _equal(out, np.cos(val), -np.sin(val) * 0)


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [-2, 4.2])
def test_cos_univariate(val, der):
    x = ad.Dual(val, der)
    out = ad.cos(x)
    assert _equal(out, np.cos(val), -np.sin(val) * der)


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize(
    "der",
    [np.array([-3.4, 6, 2]), np.array([-1, 24.2, 6])])
def test_cos_multivariate(val, der):
    x = ad.Dual(val, der)
    out = ad.cos(x)
    assert _equal(out, np.cos(val), -np.sin(val) * der)


@pytest.mark.parametrize("val", [1, -6.2])
def test_tan_constant(val):
    x = ad.Dual.constant(val)
    out = ad.tan(x)
    assert _equal(out, np.tan(val), 0 / (np.cos(val)**2))


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [-2, 4.2])
def test_tan_univariate(val, der):
    x = ad.Dual(val, der)
    out = ad.tan(x)
    assert _equal(out, np.tan(val), der / (np.cos(val)**2))


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([-1, 24.2])])
def test_tan_multivariate(val, der):
    x = ad.Dual(val, der)
    out = ad.tan(x)
    assert _equal(out, np.tan(val), der / (np.cos(val)**2))


def test_tan_derivative_undefined():
    x = ad.Dual(np.pi / 2)
    with pytest.raises(ValueError):
        ad.tan(x)


@pytest.mark.parametrize("val", [1, 2])
def test_log_constant(val):
    x = ad.Dual.constant(val)
    out = ad.log(x)
    assert _equal(out, np.log(val), 0)


@pytest.mark.parametrize("val", [0.7, 64])
@pytest.mark.parametrize("der", [-2, 4.2])
def test_log_univariate(val, der):
    x = ad.Dual(val, der)
    out = ad.log(x)
    assert _equal(out, np.log(val), der / val)


@pytest.mark.parametrize("val", [0.7, 64])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([-1, 24.2])])
def test_log_multivariate(val, der):
    x = ad.Dual(val, der)
    out = ad.log(x)
    assert _equal(out, np.log(val), der / val)


@pytest.mark.parametrize("val", [0, -2.4, -11])
def test_log_undefined(val):
    x = ad.Dual(val)
    with pytest.raises(ValueError):
        ad.log(x)


@pytest.mark.parametrize("val", [1, -6.2])
def test_exp_constant(val):
    x = ad.Dual.constant(val)
    out = ad.exp(x)
    assert _equal(out, np.exp(val), 0)


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [-2, 4.2])
def test_exp_univariate(val, der):
    x = ad.Dual(val, der)
    out = ad.exp(x)
    assert _equal(out, np.exp(val), der * np.exp(val))


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([-1, 24.2])])
def test_exp_multivariate(val, der):
    x = ad.Dual(val, der)
    out = ad.exp(x)
    assert _equal(out, np.exp(val), der * np.exp(val))


@pytest.mark.parametrize("val", [1, 6.2])
def test_sqrt_constant(val):
    x = ad.Dual.constant(val)
    out = ad.sqrt(x)
    assert _equal(out, np.sqrt(val), 0)


@pytest.mark.parametrize("val", [0.7, 64])
@pytest.mark.parametrize("der", [-2, 4.2])
def test_sqrt_univariate(val, der):
    x = ad.Dual(val, der)
    out = ad.sqrt(x)
    assert _equal(out, np.sqrt(val), 0.5 / np.sqrt(val) * der)


@pytest.mark.parametrize("val", [0.7, 64])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([-1, 24.2])])
def test_sqrt_multivariate(val, der):
    x = ad.Dual(val, der)
    out = ad.sqrt(x)
    assert _equal(out, np.sqrt(val), 0.5 / np.sqrt(val) * der)


@pytest.mark.parametrize("val", [-2.4, -11])
def test_sqrt_undefined(val):
    x = ad.Dual(val)
    with pytest.raises(ValueError):
        ad.sqrt(x)
        
@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [-2, 4.2])
def test_sinh_univariate(val, der):
    x = ad.Dual(val, der)
    out = ad.sinh(x)
    out_val = np.sinh(val)
    out_der = np.cosh(val) * der
    assert _equal(out, out_val, out_der)

@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([-1, 24.2])])
def test_sinh_multivariate(val, der):
    x = ad.Dual(val, der)
    out = ad.sinh(x)
    out_val = np.sinh(val)
    out_der = np.cosh(val) * der
    assert _equal(out, out_val, out_der)
    
@pytest.mark.parametrize("val", [1, 6.2])
def test_sinh_constant(val):
    x = ad.Dual.constant(val)
    out = ad.sinh(x)
    assert _equal(out, np.sinh(val), 0)
    
@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [-2, 4.2])
def test_cosh_univariate(val, der):
    x = ad.Dual(val, der)
    out = ad.cosh(x)
    out_val = np.cosh(val)
    out_der = np.sinh(val) * der
    assert _equal(out, out_val, out_der)

@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([-1, 24.2])])
def test_cosh_multivariate(val, der):
    x = ad.Dual(val, der)
    out = ad.cosh(x)
    out_val = np.cosh(val)
    out_der = np.sinh(val) * der
    assert _equal(out, out_val, out_der)
    
@pytest.mark.parametrize("val", [1, 6.2])
def test_cosh_constant(val):
    x = ad.Dual.constant(val)
    out = ad.cosh(x)
    assert _equal(out, np.cosh(val), 0)
    
@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [-2, 4.2])
def test_tanh_univariate(val, der):
    x = ad.Dual(val, der)
    out = ad.tanh(x)
    out_val = np.tanh(val)
    out_der = (1 - np.tanh(val)**2) * der
    assert _equal(out, out_val, out_der)

@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([-1, 24.2])])
def test_tanh_multivariate(val, der):
    x = ad.Dual(val, der)
    out = ad.tanh(x)
    out_val = np.tanh(val)
    out_der = (1 - np.tanh(val)**2) * der
    assert _equal(out, out_val, out_der)
    
@pytest.mark.parametrize("val", [1, 6.2])
def test_tanh_constant(val):
    x = ad.Dual.constant(val)
    out = ad.tanh(x)
    assert _equal(out, np.tanh(val), 0)


@pytest.mark.parametrize("val", [0.5,0.1,-0.1,-0.99,0])
def test_arcsin_constant(val):
    x = ad.Dual.constant(val)
    out = ad.arcsin(x)
    assert _equal(out, np.arcsin(val),0)


@pytest.mark.parametrize("val", [0.5,0.1,-0.1,-0.99,0])
@pytest.mark.parametrize("der", [2, 0, -1, 25.3, -19.1])
def test_arcsin_univariate(val,der):
    x = ad.Dual(val,der)
    out = ad.arcsin(x)
    assert _equal(out, np.arcsin(val), der/(np.sqrt(1-val**2)))


@pytest.mark.parametrize("val", [0.1,-0.1,-0.99,0])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([1.2, -22]),np.array([-1, 24.2]),np.array([0, 4.2])])
def test_arcsin_multivariate(val,der):
    x = ad.Dual(val,der)
    out = ad.arcsin(x)
    assert _equal(out, np.arcsin(val), der/(np.sqrt(1-val**2)))

@pytest.mark.parametrize("val",[1.0001,-1.0001,1.0001,-1.0001])
@pytest.mark.parametrize("der", [1,0,np.array([-1, 24.2]),np.array([0, 4.2])])
def test_arcsin_undefined(val,der):
    x = ad.Dual(val,der)
    with pytest.raises(ValueError):
        ad.arcsin(x)

@pytest.mark.parametrize("val", [0.5,0.1,-0.1,-0.99,0])
def test_arcsin_number(val):
    out = ad.arcsin(val)
    assert pytest.approx(out,np.arcsin(val))

@pytest.mark.parametrize("val", [0.5,0.1,-0.1,-0.99,0])
def test_arccos_constant(val):
    x = ad.Dual.constant(val)
    out = ad.arccos(x)
    assert _equal(out,np.arccos(val),0)


@pytest.mark.parametrize("val", [0.5,0.1,-0.1,-0.99,0])
@pytest.mark.parametrize("der", [2, 0, -1, 25.3, -19.1])
def test_arccos_univariate(val,der):
    x = ad.Dual(val,der)
    out = ad.arccos(x)
    assert _equal(out,np.arccos(val),-der/(np.sqrt(1 - val**2)))


@pytest.mark.parametrize("val", [0.1,-0.1,-0.99,0])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([1.2, -22]),np.array([-1, 24.2]),np.array([0, 4.2])])
def test_arccos_multivariate(val,der):
    x = ad.Dual(val,der)
    out = ad.arccos(x)
    assert _equal(out,np.arccos(val),-der/(np.sqrt(1 - val**2)))

@pytest.mark.parametrize("val",[1.0001,-1.0001,1.0001,-1.0001])
@pytest.mark.parametrize("der", [1,0,np.array([-1, 24.2]),np.array([0, 4.2])])
def test_arccos_undefined(val,der):
    x = ad.Dual(val,der)
    with pytest.raises(ValueError):
        ad.arccos(x)

@pytest.mark.parametrize("val", [0.5,0.1,-0.1,-0.99,0])
def test_arccos_number(val):
    out = ad.arccos(val)
    assert pytest.approx(out,np.arccos(val))

@pytest.mark.parametrize("val", [0.7, 64, -0.3, -10, 11.4])
def test_arctan_constant(val):
    x = ad.Dual.constant(val)
    out = ad.arctan(x)
    assert _equal(out,np.arctan(val),0)


@pytest.mark.parametrize("val", [0.7, 64, -0.3, -10, 11.4])
@pytest.mark.parametrize("der", [2, 0, -1, 25.3, -19.1])
def test_arctan_univariate(val,der):
    x = ad.Dual(val,der)
    out = ad.arctan(x)
    assert _equal(out,np.arctan(val),der/(1+val**2))

@pytest.mark.parametrize("val", [0.7, -0.3, -10, 11.4])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([1.2, -22]),np.array([-1, 24.2]),np.array([0, 4.2])])
def test_arctan_multivariate(val,der):
    x = ad.Dual(val,der)
    out = ad.arctan(x)
    assert _equal(out,np.arctan(val),der/(1+val**2))

@pytest.mark.parametrize("val", [0.7, 64, -0.3, -10, 11.4])
def test_arctan_number(val):
    out = ad.arctan(val)
    assert pytest.approx(out,np.arctan(val))


@pytest.mark.parametrize("val", [0.7, 64, -0.5, 10, -10])
def test_logistic_constant(val):
    g = lambda z: 1 / (1 + np.exp(-z))
    x = ad.Dual.constant(val)
    out = ad.logistic(x)
    assert _equal(out,g(val),0)

@pytest.mark.parametrize("val", [0.7, 64, -0.5, 10, -10])
def test_logistic_numbers(val):
    g = lambda z: 1 / (1 + np.exp(-z))
    out = ad.logistic(val)
    assert pytest.approx(out,g(val))

@pytest.mark.parametrize("val", [0.7, 64, -0.3, -10, 11.4])
@pytest.mark.parametrize("der", [2, 0, -1, 25.3, -19.1])
def test_logistic_univariate(val,der):
    g = lambda z: 1 / (1 + np.exp(-z))
    x = ad.Dual(val,der)
    out = ad.logistic(x)
    assert _equal(out,g(val),der*g(val)*(1-g(val)))


@pytest.mark.parametrize("val", [0.7, -0.3, -10, 11.4])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([1.2, -22]),np.array([-1, 24.2]),np.array([0, 4.2])])
def test_logistic_multivariate(val,der):
    g = lambda z: 1 / (1 + np.exp(-z))
    x = ad.Dual(val,der)
    out = ad.logistic(x)
    assert _equal(out,g(val),der*g(val)*(1-g(val)))
