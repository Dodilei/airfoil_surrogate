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

    return comb * (t ** k) * ( (1 - t) ** (n - k))
    
# %% BEZIER CURVE
def bezier_curve(control_points, degree):
    """Compute the Bezier curve."""
    return lambda t: np.dot(bernstein_poly(degree, t).T, control_points)

# %% VISUALIZATION
def plot_bezier_curve(control_points, degree, num_points=100):
    """Plot the Bezier curve along with its control points."""
    import matplotlib.pyplot as plt

    t_values = np.linspace(0, 1, num_points)
    curve_points = bezier_curve(control_points, degree)(t_values)

    plt.plot(curve_points[:, 0], curve_points[:, 1], label='Bezier Curve')
    plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Control Points')
    plt.title('Bezier Curve of Degree {}'.format(degree))
    plt.legend()
    plt.grid()
    plt.show()

# %% EXAMPLE USAGE
if __name__ == "__main__":
    # Define control points
    control_points = np.array([[0, 0], [1, 2], [3, 3], [4, 0]])
    degree = len(control_points) - 1

    # Plot the Bezier curve
    plot_bezier_curve(control_points, degree)