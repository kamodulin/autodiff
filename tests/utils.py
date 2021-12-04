import numpy as np

import autodiff as ad

import autodiff.reverse as adr


def _equal(x, val, der, eval_der=None):
    if eval_der is None:
        return np.isclose(x.val, val) and np.isclose(x.der, der).all()
    else:
        return np.isclose(x.val, val) and np.isclose(eval_der, der).all()


def _compare(comparison, val, der):
    x = ad.Dual(*comparison)
    return _equal(x, val, der)


def _compare_node(comparison, val, der, eval_der):
    x = adr.Node(comparison)
    return _equal(x,val,der,eval_der)
