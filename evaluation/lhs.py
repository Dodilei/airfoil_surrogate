from scipy.stats import qmc


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
