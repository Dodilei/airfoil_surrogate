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

    beta = np.linspace(0, np.pi, num_points // 2)
    t = 0.5 * (1 - np.cos(beta))  # Cosine spacing

    yu, yl = airfoil(t)

    x_coords = t
    upper_surface = np.vstack((x_coords[::-1], yu[::-1])).T
    lower_surface = np.vstack((x_coords, yl)).T

    return np.vstack((upper_surface[:-1], lower_surface))


def parse_airfoil_coordinates(xy_coords, refit=False):
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

    if refit:
        n_points = refit
    else:
        n_points = max(len(x1), len(x2))

    beta = np.linspace(0, np.pi, n_points)
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


def fit_airfoil_shape_ct(airfoil_coords, order_camber, order_thick, visualize=False):
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


def fit_airfoil_shape_kulfan(airfoil_coords, order, visualize=False):
    """Fit an airfoil shape to given coordinates using CST Bezier representation."""

    x_base, yu, yl = airfoil_coords

    # --- Step 2: Handle TE Thickness ---
    te_thick = yu[-1] - yl[-1]
    # Remove linear wedge term so we fit pure CST
    yu_clean = yu - x_base * (te_thick / 2)
    yl_clean = yl - x_base * (-te_thick / 2)

    # --- Step 3: Build Matrix ---

    # Matrix K has shape (n_points, n_weights)
    K = cst_bezier_matrix(np.ones(order + 1), lambda t: np.sqrt(t) * (1 - t))(x_base).T

    # --- Step 4: Determine Best Shared A0 (Leading Edge) ---
    # We solve strictly for the first point (limit as x->0).
    # Or simpler: Fit independent, then average magnitudes.
    Au_temp, _, _, _ = np.linalg.lstsq(K, yu_clean, rcond=None)
    Al_temp, _, _, _ = np.linalg.lstsq(K, yl_clean, rcond=None)

    # Calculate shared A0 (Magnitude)
    # This is the "Constraint" we must enforce.
    a0_shared = (abs(Au_temp[0]) + abs(Al_temp[0])) / 2

    # --- Step 5: The "Refit" (Constraint Enforcement) ---
    # We know A0. The equation is: Y = A0*Base0 + A1*Base1 + ...
    # So: (Y - A0*Base0) = A1*Base1 + ...

    # Isolate the "Fixed" part (First column of K scaled by A0)
    fixed_term_u = K[:, 0] * a0_shared
    fixed_term_l = K[:, 0] * -a0_shared  # Lower surface has negative A0

    # Subtract fixed part from target
    yu_residual = yu_clean - fixed_term_u
    yl_residual = yl_clean - fixed_term_l

    # Matrix for remaining weights (Columns 1 to 8)
    K_remaining = K[:, 1:]

    # Solve for A1...A8
    Au_rest, _, _, _ = np.linalg.lstsq(K_remaining, yu_residual, rcond=None)
    Al_rest, _, _, _ = np.linalg.lstsq(K_remaining, yl_residual, rcond=None)

    if visualize:
        import matplotlib.pyplot as plt

        y_upper_fit = np.hstack((a0_shared, Au_rest)) @ K.T
        y_lower_fit = np.hstack((-a0_shared, Al_rest)) @ K.T

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x_base, yu, "k-", label="Original S1223 (Target)")
        plt.plot(x_base, yl, "k-")
        plt.plot(x_base, y_upper_fit, "r--", linewidth=2, label="CST Fit (Result)")
        plt.plot(x_base, y_lower_fit, "r--", linewidth=2)
        plt.axis("equal")
        plt.legend()
        plt.title(f"Fitting S1223 with CST (Upper Order {order}, Lower Order {order})")
        plt.grid(True)
        plt.show()

    return a0_shared, Au_rest, Al_rest, te_thick


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
