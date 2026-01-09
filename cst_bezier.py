# %% INIT
import numpy as np

# %% COMBINATION


# n!/k!(n-k)!
def combinations(n):
    """Compute the combination of n and k (n choose k) for all k."""
    k = np.arange(n + 1)
    k_fact = np.ones(n + 1)
    k_fact[1:] = np.cumprod(k[1:])
    n_fact = k_fact[-1]
    n_k_fact = k_fact[::-1]

    return n_fact / (k_fact * n_k_fact)


# %% BERNSTEIN POLYNOMIALS
def bernstein_poly(n, t):
    """Compute the Bernstein polynomials of degree n at parameter t."""
    k = np.arange(n + 1)[:, np.newaxis]
    comb = combinations(n)[:, np.newaxis]
    t = np.asarray(t)[np.newaxis, :]

    return comb * (t**k) * ((1 - t) ** (n - k))


# %% CST BEZIER CURVE
def cst_bezier(coefs, class_function, degree):
    """Compute a CST Bezier curve given coefficients, a class function, and degree."""

    return lambda t: class_function(t) * np.dot(bernstein_poly(degree, t).T, coefs)
