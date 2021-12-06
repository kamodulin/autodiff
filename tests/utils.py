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
    return (np.isclose(x.val, val)
            and np.isclose(eval_der, der).all()) if eval_der is not None else (
                np.isclose(x.val, val) and eval_der == None)


def fdn(g, X, epi=1e-6):
    """
    Gradient checking function with finite difference method.

    Parameters
    ----------
    g : function
        Function to be checked.
    X : array_like
        Input array.
    epi : float, optional

    Returns
    -------
    out : list
        List of gradients.
    """
    f = lambda x: g(*x)
    mat = np.eye(len(X)) * epi
    X_plus = X + mat
    X_minus = X - mat
    return (np.apply_along_axis(f, 1, X_plus) -
            np.apply_along_axis(f, 1, X_minus)) / (2 * epi)
