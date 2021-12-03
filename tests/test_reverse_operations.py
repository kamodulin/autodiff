import numpy as np
import pytest

import autodiff.reverse as ad

from utils import _equal


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
    x._add_child(der, child)
    out = ad.sin(x)
    der = np.cos(val) + der
    eval_der = np.array([x.grad()])
    assert _equal(out, np.sin(val), der, eval_der)
