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

    return (np.isclose(x.val, val) and np.isclose(eval_der, der).all()) if eval_der is not None else (np.isclose(x.val, val) and eval_der == None)

def fdn(g, x, epi = 1e-4):
    f = lambda y: g(*y)
    d = len(x)
    mat = np.eye(d) * epi
    xplus =  x + mat
    xminus = x - mat
    return (np.apply_along_axis(f,1,xplus) - np.apply_along_axis(f,1,xminus))/(2*epi)