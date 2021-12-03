import numpy as np

import autodiff as ad

import autodiff.reverse as adr

def _equal(x, val, der, eval_der = None):
    if eval_der is None:
        return x.val == val and np.all(x.der == der)
    else:
        return x.val == val and np.all(eval_der == der)

def _compare(comparison, val, der):
    x = ad.Dual(*comparison)
    return _equal(x, val, der)

def _compare_node(comparison, val, der, eval_der):
    x = adr.Node(comparison)
    return _equal(x,val,der,eval_der)