import numpy as np
import pytest

import autodiff as ad

from utils import _equal, fdn

@pytest.mark.parametrize("val1", [0.7, -20, 100])
@pytest.mark.parametrize("val2", [0.7, 64, 200])
def test_trig_integration(val1,val2):
	x, y = ad.Dual.from_array([val1,val2])
	f = (ad.sin(x) + ad.cos(y) - ad.tan(x * y)) / x**2

	npf = lambda x, y: (np.sin(x) + np.cos(y) - np.tan(x * y)) / x**2
	out = npf(val1,val2)
	der = fdn(npf,[val1,val2])
	assert _equal(f, out, der)



@pytest.mark.parametrize("val1", [0.7, -5, -4.2, 8, 6])
@pytest.mark.parametrize("val2", [0.7, 5.9, -0.3, -4, 8])
def test_hyper_integration(val1,val2):
	x, y = ad.Dual.from_array([val1,val2])
	f = (ad.sinh(x) + ad.cosh(y) - ad.tanh(x)) / x**2

	npf = lambda x, y: (np.sinh(x) + np.cosh(y) - np.tanh(x)) / x**2
	out = npf(val1,val2)
	der = fdn(npf,[val1,val2])
	assert _equal(f, out, der)


@pytest.mark.parametrize("val1", [0.7, 0.2, -0.2, 0.8, 0.6])
@pytest.mark.parametrize("val2", [0.7, 0.9, -0.3, -0.4, 0.8])
def test_inversetrig_integration(val1, val2):
	x, y = ad.Dual.from_array([val1,val2])
	f = (ad.arcsin(x) + ad.arccos(y) - ad.arctan(x)) / x**2

	npf = lambda x, y: (np.arcsin(x) + np.arccos(y) - np.arctan(x)) / x**2
	out = npf(val1,val2)
	der = fdn(npf,[val1,val2])
	assert _equal(f, out, der)


@pytest.mark.parametrize("val1", [0.7, -5, -4.2, 8, 6])
@pytest.mark.parametrize("val2", [0.7, 5.9, -0.3, -4, 8])
@pytest.mark.parametrize("val3", [1.5, 4, 0.2, 8, 6])
def test_misc_integration(val1, val2,val3):
	x, y, z= ad.Dual.from_array([val1,val2,val3])
	f = (ad.logistic(x) + ad.sqrt(ad.exp(y)) - ad.log(z)) / x**2 + ad.log(z, base=10)

	g = lambda x: 1 / (1 + np.exp(-x))
	npf = lambda x, y, z: (g(x) + np.sqrt(np.exp(y)) - np.log(z)) / x**2 + np.log10(z)
	out = npf(val1,val2,val3)
	der = fdn(npf,[val1,val2,val3])
	assert _equal(f, out, der)