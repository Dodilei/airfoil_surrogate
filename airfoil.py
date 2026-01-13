import numpy as np
from cst_bezier import cst_bezier, cst_bezier_matrix


def camber_line(coefs):
    """Compute the camber line using CST Bezier representation."""
    # Class function for camber line (N1 = 1.0, N2 = 1.0)
    class_function = lambda t: t**1.0 * (1 - t) ** 1.0
    return cst_bezier(coefs, class_function)


def thickness_distribution(coefs):
    """Compute the thickness distribution using CST Bezier representation."""
    # Class function for thickness distribution (N1 = 0.5, N2 = 1.0)
    class_function = lambda t: t**0.5 * (1 - t) ** 1.0
    return cst_bezier(coefs, class_function)


def airfoil_shape(camber_coefs, thickness_coefs, te_thickness=0.0):
    """Compute the airfoil shape given camber and thickness coefficients."""
    camber = camber_line(camber_coefs)
    thickness = thickness_distribution(thickness_coefs)

    def airfoil(t):
        yc = camber(t)
        yt = thickness(t)

        # Upper surface
        yu = yc + yt / 2 + te_thickness / 2 * t
        # Lower surface
        yl = yc - yt / 2 - te_thickness / 2 * t

        return yu, yl

    return airfoil


def airfoil_coordinates(airfoil, num_points=100):
    """Generate airfoil coordinates with cosine spacing."""

    beta = np.linspace(0, np.pi, num_points)
    t = 0.5 * (1 - np.cos(beta))  # Cosine spacing

    yu, yl = airfoil(t)

    x_coords = t
    upper_surface = np.vstack((x_coords, yu)).T
    lower_surface = np.vstack((x_coords[::-1], yl[::-1])).T

    return np.vstack((upper_surface, lower_surface))


def parse_airfoil_coordinates(xy_coords):
    """Parse airfoil coordinates into upper and lower surfaces."""
    # Assuming coords is an array of shape (2, N)
    mask_side = np.diff(xy_coords[0, :], prepend=0) < 0
    mask_side[0] = mask_side[1]

    y1 = xy_coords[1, mask_side][::-1]
    y2 = xy_coords[1, ~mask_side]
    x1 = xy_coords[0, mask_side][::-1]
    x2 = xy_coords[0, ~mask_side]

    xmax = np.max(xy_coords[0, :])
    xmin = np.min(xy_coords[0, :])

    x1_norm = (x1 - xmin) / (xmax - xmin)
    x2_norm = (x2 - xmin) / (xmax - xmin)

    beta = np.linspace(0, np.pi, len(x1_norm))
    t1 = 0.5 * (1 - np.cos(beta))  # Cosine spacing

    y1_refit = np.interp(t1, x1_norm, y1)
    y2_refit = np.interp(t1, x2_norm, y2)

    if y1_refit.sum() > y2_refit.sum():
        yu = y1_refit
        yl = y2_refit
    else:
        yu = y2_refit
        yl = y1_refit

    return t1, yu, yl


def fit_airfoil_shape(airfoil_coords, order_camber, order_thick, visualize=False):
    """Fit an airfoil shape to given coordinates using CST Bezier representation."""

    x_base, yu, yl = airfoil_coords

    yc_target = (yu + yl) / 2
    yt_target = yu - yl

    k_camber = cst_bezier_matrix(
        np.ones(order_camber + 1), lambda t: t**1.0 * (1 - t) ** 1.0
    )(x_base)
    A_camber, _, _, _ = np.linalg.lstsq(k_camber.T, yc_target, rcond=None)

    k_thick = cst_bezier_matrix(
        np.ones(order_thick + 1), lambda t: t**0.5 * (1 - t) ** 1.0
    )(x_base)
    A_thick, _, _, _ = np.linalg.lstsq(k_thick.T, yt_target, rcond=None)

    if visualize:
        import matplotlib.pyplot as plt

        y_camber_fit = A_camber @ k_camber
        y_thick_fit = A_thick @ k_thick
        y_up_fit = y_camber_fit + y_thick_fit / 2
        y_low_fit = y_camber_fit - y_thick_fit / 2

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x_base, yu, "k-", label="Original S1223 (Target)")
        plt.plot(x_base, yl, "k-")
        plt.plot(x_base, y_up_fit, "r--", linewidth=2, label="CST Fit (Result)")
        plt.plot(x_base, y_low_fit, "r--", linewidth=2)
        plt.axis("equal")
        plt.legend()
        plt.title(
            f"Fitting S1223 with CST (Camber Order {order_camber}, Thick Order {order_thick})"
        )
        plt.grid(True)
        plt.show()

    return A_camber, A_thick


# Example usage:
if __name__ == "__main__":
    # Define camber and thickness coefficients
    camber_coefs = np.array([0.0, 0.1, 0.0])  # Example camber coefficients
    thickness_coefs = np.array([0.2, 0.3, 0.2])  # Example thickness coefficients
    degree = len(camber_coefs) - 1
    te_thickness = 0.02  # Example trailing edge thickness

    # Create airfoil shape function
    airfoil = airfoil_shape(camber_coefs, thickness_coefs, te_thickness)

    # Generate airfoil coordinates
    coords = airfoil_coordinates(airfoil, num_points=100)

    # Plot the coordinates as needed
    import matplotlib.pyplot as plt

    plt.plot(coords[:, 0], coords[:, 1], "-o", ms=2, lw=1)
    plt.axis("equal")
    plt.title("Airfoil Shape")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
