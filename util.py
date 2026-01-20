import numpy as np
from scipy.stats import qmc


def mellowmax(x, alpha=100):
    """
    Calculates Mellowmax.
    alpha > 0  => Soft Maximum (approximates max)
    alpha < 0  => Soft Minimum (approximates min)
    """
    x = np.array(x)

    # 1. Conditioning to prevent overflow
    # For Max (alpha > 0): shift by subtracting actual max
    # For Min (alpha < 0): shift by subtracting actual min
    if alpha > 0:
        shift = np.max(x)
    else:
        shift = np.min(x)

    # 2. The Stable Formula
    # MM(x) = shift + (1/alpha) * ln( mean( exp(alpha * (x - shift)) ) )
    n = len(x)
    scaled_exp_sum = np.sum(np.exp(alpha * (x - shift)))

    # Note: We use n (mean) inside the log, unlike KS/LSE which use sum.
    return shift + (np.log(scaled_exp_sum / n) / alpha)


def generate_lhs(bounds, n_samples):
    """
    Generates a Latin Hypercube Sample of size n_samples
    within the provided bounds.

    Parameters:
        bounds (np.ndarray): (M, 2) array where col 0 is min and col 1 is max.
        n_samples (int): Total number of cases to generate.

    Returns:
        np.ndarray: (n_samples, M) array of parameters.
    """
    # 1. Initialize the Latin Hypercube sampler
    # d = dimensionality (number of parameters/rows in bounds)
    dim = len(bounds)
    sampler = qmc.LatinHypercube(d=dim, optimization="random-cd")

    # 2. Generate samples in the unit hypercube [0, 1]
    # Shape will be (n_samples, dim)
    unit_sample = sampler.random(n=n_samples)

    # 3. Scale samples from [0, 1] to actual bounds
    # l_bounds = lower limits, u_bounds = upper limits
    l_bounds = bounds[:, 0]
    u_bounds = bounds[:, 1]

    # qmc.scale automatically maps the [0,1] samples to [min, max]
    lhs_samples = qmc.scale(unit_sample, l_bounds, u_bounds)

    return lhs_samples
