import numpy as np

import autodiff as ad


def _equal(x, val, der):
    return x.val == val and np.all(x.der == der)


def _compare(comparison, val, der):
    x = ad.Dual(*comparison)
    return _equal(x, val, der)
