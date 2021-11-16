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
    assert _equal(out, np.exp(val), der*np.exp(val))


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([-1, 24.2])])
def test_exp_multivariate(val, der):
    x = ad.Dual(val, der)
    out = ad.exp(x)
    assert _equal(out, np.exp(val), der*np.exp(val))