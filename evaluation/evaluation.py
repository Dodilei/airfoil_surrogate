import numpy as np
import aerosandbox
from geometry.airfoil import morph_airfoil_ct, ct2coords

from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import PchipInterpolator


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


class AirfoilVarEvaluator(object):
    def __init__(
        self,
        base_foil_ct,
        alpha_sim,
        alpha_target_specs,
        slope_window_size=6.0,
    ):
        self.alpha_sim = alpha_sim
        self.alpha_target, at_step = np.linspace(*alpha_target_specs, retstep=True)
        self.base_foil_ct = base_foil_ct

        self.conv_threshold = len(alpha_sim) // 2

        self.slope_calculator = ScanningSlopeFinder(
            self.alpha_target,
            window_size_deg=slope_window_size,
            alpha_bounds=[-1.5, 12.0],
        )

        self.xfoil_path = "xfoil"
        self.xfoil_max_iter = 50
        self.xfoil_timeout = 25

    def xfoil_config_set(self, path, max_iter=50, timeout=25):
        self.xfoil_path = path
        self.xfoil_max_iter = max_iter
        self.xfoil_timeout = timeout

    def evaluate_variation(self, X):
        reynolds = int(np.exp(X[0]))
        variation_specs = X[1:]

        coords = ct2coords(morph_airfoil_ct(self.base_foil_ct, *variation_specs))

        cl, cd, cm, conv_alpha = self.run_xfoil(
            coords,
            reynolds,
        )

        if len(conv_alpha) < self.conv_threshold:
            return [np.nan] * 8

        return self.extract_parameters(cl, cd, cm, conv_alpha)

    def evaluate_airfoil_visual(self, X):
        reynolds = int(np.exp(X[0]))
        variation_specs = X[1:]

        coords = ct2coords(morph_airfoil_ct(self.base_foil_ct, *variation_specs))

        cl, cd, cm, conv_alpha = self.run_xfoil(
            coords,
            reynolds,
        )

        cl_mapper = PchipInterpolator(conv_alpha, cl)

        cl_full = cl_mapper(self.alpha_target)

        (
            cl_max,
            cd_at_clmax,
            cm_at_clmax,
            cl_at_cdmin,
            cd_min,
            cm_at_cdmin,
            dclda,
            a0L,
        ) = self.extract_parameters(cl, cd, cm, conv_alpha)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(16, 16))

        ax1, ax2, ax3, ax4 = axes.flatten()

        specs = X[1:]

        t1, yc0, yt0 = self.base_foil_ct

        x_base, yc, yt = morph_airfoil_ct((t1, yc0, yt0), *specs)
        coords = ct2coords((x_base, yc, yt))

        upper0 = np.vstack((x_base, yc0 + 0.5 * yt0)).T[::-1]
        lower0 = np.vstack((x_base, yc0 - 0.5 * yt0)).T
        coords0 = np.vstack((upper0, lower0[1:, :]))

        ax1.plot(coords0[:, 0], coords0[:, 1], "k-", linewidth=3)
        ax1.plot(x_base, yc0, "k--", linewidth=2)
        ax1.plot(coords[:, 0], coords[:, 1], "r--", linewidth=2)
        ax1.plot(x_base, yc, "r-.", linewidth=1)

        ax2.plot(conv_alpha, cl, "o-", label="CL vs Alpha")
        ax2.plot(self.alpha_target, cl_full, "k-.")

        idx_half_range = len(self.alpha_target) // 2
        mid_window_size = int(0.6 * idx_half_range)

        range_half1 = self.alpha_target[[0, idx_half_range - 1]]
        range_half2 = self.alpha_target[[idx_half_range, -1]]

        ax2.plot(range_half2, [cl_max, cl_max], "r--", label="CL Max")
        ax2.plot(
            range_half1,
            [cl_at_cdmin, cl_at_cdmin],
            "r--",
            label="CL CDmin",
        )

        dclda_window = np.array(
            [
                self.alpha_target[idx_half_range - mid_window_size],
                self.alpha_target[idx_half_range + mid_window_size],
            ]
        )
        ax2.plot(
            dclda_window,
            dclda * (dclda_window - a0L),
            "g-.",
            label="dCl/dAlpha",
        )

        ax3.plot(conv_alpha, cd, "o-", label="CD vs Alpha")

        ax3.plot(range_half1, [cd_min, cd_min], "r--", label="CD Min")
        ax3.plot(
            range_half2,
            [cd_at_clmax, cd_at_clmax],
            "r--",
            label="CD CLmax",
        )

        ax4.plot(conv_alpha, cm, "o-", label="Cm vs Alpha")

        ax4.plot(
            range_half2,
            [cm_at_clmax, cm_at_clmax],
            "r--",
            label="Cm CLmax",
        )
        ax4.plot(
            range_half1,
            [cm_at_cdmin, cm_at_cdmin],
            "r--",
            label="Cm CDmin",
        )

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        ax1.set_title("Airfoil Shape Comparison")
        ax1.set_xlabel("x/c")

        ax2.set_title("Lift Coefficient vs Angle of Attack")
        ax2.set_xlabel("Angle of Attack (degrees)")
        ax2.set_ylabel("Cl")

        ax3.set_title("Drag Coefficient vs Angle of Attack")
        ax3.set_xlabel("Angle of Attack (degrees)")
        ax3.set_ylabel("Cd")

        ax4.set_title("Moment Coefficient vs Angle of Attack")
        ax4.set_xlabel("Angle of Attack (degrees)")
        ax4.set_ylabel("Cm")

        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

    def run_xfoil(self, coords, reynolds):
        airfoil = aerosandbox.Airfoil(coordinates=coords)

        xf = aerosandbox.XFoil(
            airfoil=airfoil,
            Re=reynolds,
            xfoil_command=self.xfoil_path,
            max_iter=self.xfoil_max_iter,
            timeout=self.xfoil_timeout,
            xfoil_repanel_n_points=150,
        )
        result = xf.alpha(self.alpha_sim)

        conv_alpha = np.array(result["alpha"])

        cl = np.array(result["CL"])
        cd = np.array(result["CD"])
        cm = np.array(result["CM"])

        return cl, cd, cm, conv_alpha

    def extract_parameters(self, cl, cd, cm, conv_alpha):
        cl_max = mellowmax(cl, alpha=100)
        cl95_mask = cl > (0.95 * cl_max)

        cl95_alpha = conv_alpha[cl95_mask]
        cl95_diff = -np.abs(cl[cl95_mask] - cl_max)

        diff_weight = (cl95_diff - 2 * cl95_diff.min()) / (
            cl95_diff.max() - 2 * cl95_diff.min()
        )
        alpha_weigth = (cl95_alpha - 1.1 * cl95_alpha.max()) / (
            cl95_alpha.min() - 1.1 * cl95_alpha.max()
        )

        cl95_weight = (0.5 + 0.5 * diff_weight) * alpha_weigth
        cl95_weight /= cl95_weight.sum()

        cd_at_clmax = (cl95_weight * cd[cl95_mask]).sum()
        cm_at_clmax = (cl95_weight * cm[cl95_mask]).sum()

        cd_min = mellowmax(cd, alpha=-1000)
        cd05_mask = cd < (1.05 * cd_min)
        cl_at_cdmin = cl[cd05_mask].mean()
        cm_at_cdmin = cm[cd05_mask].mean()

        cl_mapper = PchipInterpolator(conv_alpha, cl)

        cl_full = cl_mapper(self.alpha_target)

        dclda, cl0 = self.slope_calculator.calculate(cl_full)

        a0L = -cl0 / dclda

        return (
            cl_max,
            cd_at_clmax,
            cm_at_clmax,
            cl_at_cdmin,
            cd_min,
            cm_at_cdmin,
            dclda,
            a0L,
        )


class ScanningSlopeFinder:
    def __init__(
        self,
        alpha_grid,
        window_size_deg=5.0,
        alpha_bounds=(-5, 12),
        min_slope=0.0,
    ):
        """
        Args:
            window_size_deg (float): Range of alpha to include in the rolling window.
            alpha_bounds (tuple): (min_alpha, max_alpha) in degrees. Only scans windows
                                  within this range.
            min_slope (float): Minimum slope to consider valid (rejects stall/negative slopes).
        """

        self.alpha_grid = alpha_grid
        alpha_grid_step = np.diff(alpha_grid).mean()

        self.window_size = int(round(window_size_deg / alpha_grid_step))
        self.min_slope = min_slope
        self.alpha_bounds = np.array(alpha_bounds) - np.array(
            [0, self.window_size * alpha_grid_step]
        )

    def calculate(self, cl):
        """
        Vectorized scanning of the Cl-alpha curve.
        Returns the slope of the window with the highest R^2.
        """
        alpha = self.alpha_grid

        # Ensure we have enough points for at least one window
        if len(alpha) < self.window_size:
            raise ValueError

        # 2. Restrict Search Space
        valid_start_indices = (
            alpha[: -self.window_size + 1] >= self.alpha_bounds[0]
        ) & (alpha[: -self.window_size + 1] <= self.alpha_bounds[1])

        if not np.any(valid_start_indices):
            raise ValueError

        # 3. Create Views (Vectorized Windows)
        # Shape becomes (N_windows, window_size)
        alpha_windows = sliding_window_view(alpha, window_shape=self.window_size)
        cl_windows = sliding_window_view(cl, window_shape=self.window_size)

        # Filter windows based on the bounds determined in step 2
        # Note: sliding_window_view output length aligns with the valid_start_indices logic
        alpha_windows = alpha_windows[valid_start_indices]
        cl_windows = cl_windows[valid_start_indices]

        # 4. Vectorized Linear Regression
        # We calculate Slope and R^2 for ALL windows in parallel using matrix math.

        # Calculate means for each window
        x_bar = alpha_windows.mean(axis=1, keepdims=True)
        y_bar = cl_windows.mean(axis=1, keepdims=True)

        # Centered variables (residuals from mean)
        dx = alpha_windows - x_bar
        dy = cl_windows - y_bar

        # Sum of Squared Errors
        Sxx = np.sum(dx**2, axis=1, keepdims=True)
        Syy = np.sum(dy**2, axis=1, keepdims=True)
        Sxy = np.sum(dx * dy, axis=1, keepdims=True)

        # 4. Slope and Intercept calculation
        # m = Sxy / Sxx
        # b = y_mean - m * x_mean
        slopes = np.full(x_bar.shape, np.nan)
        intercepts = np.full(x_bar.shape, np.nan)
        r2 = np.full(x_bar.shape, -np.inf)

        valid_mask = Sxx > 1e-10
        slopes[valid_mask] = Sxy[valid_mask] / Sxx[valid_mask]
        intercepts[valid_mask] = y_bar[valid_mask] - (
            slopes[valid_mask] * x_bar[valid_mask]
        )

        # R^2 calculation
        r2_denom = Sxx * Syy
        r2_mask = valid_mask & (r2_denom > 1e-10)
        r2[r2_mask] = (Sxy[r2_mask] ** 2) / r2_denom[r2_mask]

        # 5. Physics Constraints
        valid_candidates = slopes > self.min_slope
        r2[~valid_candidates] = -np.inf

        best_idx = np.argmax(r2)

        if r2[best_idx] == -np.inf:
            return np.nan, np.nan

        return slopes.flatten()[best_idx], intercepts.flatten()[best_idx]
