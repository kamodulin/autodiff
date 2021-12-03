import numpy as np
import pytest

import autodiff.reverse as ad

from utils import _equal, _compare_node


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_add_constant(val1, val2):
    x = ad.Node.constant(val1)
    y = ad.Node.constant(val2)
    assert _equal(x + y, val1 + val2, 0, np.array([x.grad(),y.grad()]))

@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_add_variable(val1, val2):
    x = ad.Node(val1)
    y = ad.Node(val2)
    assert _equal(x + y, val1 + val2, np.array([1.0, 1.0]), np.array([x.grad(),y.grad()]))