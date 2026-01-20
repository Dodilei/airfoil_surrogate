import numpy as np
import aerosandbox
from airfoil import morph_airfoil_ct, ct2coords
from util import mellowmax


class AirfoilVarEvaluator(object):
    def __init__(
        self,
        base_foil_ct,
        alpha_sim,
        alpha_target_specs,
        slope_window_center=1.0,
        slope_window_size=5.0,
        slope_window_displacement=2.5,
    ):
        self.alpha_sim = alpha_sim
        self.alpha_target, at_step = np.linspace(*alpha_target_specs, retstep=True)
        self.base_foil_ct = base_foil_ct

        self.conv_threshold = len(alpha_sim) // 2

        slope_window_start_idx = int(
            (slope_window_center - slope_window_size / 2 - self.alpha_target[0])
            / at_step
        )
        slope_window_size_idx = int(slope_window_size / at_step)
        slope_window_displacement_idx = int(slope_window_displacement / at_step)

        self.slope_calculator = ThreeWindowSlope(
            self.alpha_target,
            center_start_idx=slope_window_start_idx,
            window_size=slope_window_size_idx,
            displacement=slope_window_displacement_idx,
        )

    def evaluate_variation(self, X, xfoil_path="xfoil.exe"):
        reynolds = int(np.exp(X[0]))
        variation_specs = X[1:]

        coords = ct2coords(morph_airfoil_ct(self.base_foil_ct, *variation_specs))

        cl, cd, cm, conv_alpha = self.run_xfoil(
            coords,
            reynolds,
            xfoil_path=xfoil_path,
        )

        return self.extract_parameters(cl, cd, cm, conv_alpha)

    def evaluate_airfoil_visual(self, X, xfoil_path="xfoil.exe"):
        reynolds = int(np.exp(X[0]))
        variation_specs = X[1:]

        coords = ct2coords(morph_airfoil_ct(self.base_foil_ct, *variation_specs))

        cl, cd, cm, conv_alpha = self.run_xfoil(
            coords,
            reynolds,
            xfoil_path=xfoil_path,
        )

        cl_full = np.interp(
            self.alpha_target, conv_alpha, cl, left=np.nan, right=np.nan
        )

        cl_max, cd_at_clmax, cm_at_clmax, cl_at_cdmin, cd_min, cm_at_cdmin, dclda = (
            self.extract_parameters(cl, cd, cm, conv_alpha)
        )

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

        dclda_window = [
            self.alpha_target[idx_half_range - mid_window_size],
            self.alpha_target[idx_half_range + mid_window_size],
        ]
        cl_in_window = [
            cl_full[idx_half_range - mid_window_size],
            cl_full[idx_half_range + mid_window_size],
        ]
        ax2.plot(
            dclda_window,
            [
                (cl_in_window[0] + cl_in_window[1]) / 2
                - dclda * (dclda_window[1] - dclda_window[0]) / 2,
                (cl_in_window[0] + cl_in_window[1]) / 2
                + dclda * (dclda_window[1] - dclda_window[0]) / 2,
            ],
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

    def run_xfoil(
        self,
        coords,
        reynolds,
        xfoil_path="xfoil.exe",
    ):
        airfoil = aerosandbox.Airfoil(coordinates=coords)

        xf = aerosandbox.XFoil(
            airfoil=airfoil,
            Re=reynolds,
            xfoil_command=xfoil_path,
            max_iter=50,
            timeout=60,
            xfoil_repanel_n_points=150,
        )
        result = xf.alpha(self.alpha_sim)

        conv_alpha = np.array(result["alpha"])

        if len(conv_alpha) < self.conv_threshold:
            return [np.nan] * 7

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

        cl_full = np.interp(
            self.alpha_target, conv_alpha, cl, left=np.nan, right=np.nan
        )
        dclda = self.slope_calculator.get_best_slope(cl_full)

        return (
            cl_max,
            cd_at_clmax,
            cm_at_clmax,
            cl_at_cdmin,
            cd_min,
            cm_at_cdmin,
            dclda,
        )


class ThreeWindowSlope:
    def __init__(self, alpha_grid, center_start_idx, window_size, displacement):
        """
        SETUP PHASE: Pre-calculates all Alpha-dependent terms.
        Runs ONCE.
        """
        self.window_size = window_size
        n = window_size

        # 1. Define the 3 sets of indices (Left, Center, Right)
        # We ensure they stay within bounds
        idx_c = center_start_idx
        idx_l = max(0, center_start_idx - displacement)
        idx_r = min(len(alpha_grid) - window_size, center_start_idx + displacement)

        self.indices = [
            slice(idx_l, idx_l + window_size),
            slice(idx_c, idx_c + window_size),
            slice(idx_r, idx_r + window_size),
        ]

        # 2. Extract Alpha segments for these 3 windows
        # Shape: (3, window_size)
        self.alphas = np.array(
            [
                alpha_grid[self.indices[0]],
                alpha_grid[self.indices[1]],
                alpha_grid[self.indices[2]],
            ]
        )

        # 3. Pre-calculate Alpha Statistics (Vectorized for 3 windows)
        # These are constant for every single execution!
        self.Sx = np.sum(self.alphas, axis=1)  # Sum of X for each window
        self.Sxx = np.sum(self.alphas**2, axis=1)  # Sum of X^2 for each window
        self.n = n

        # Denominator for Slope: n*Sxx - Sx^2
        self.denom_slope = n * self.Sxx - self.Sx**2

        # Pre-calculate terms for R2 denominator
        # We need (n*Sxx - Sx^2) which is exactly denom_slope
        self.denom_r2_part1 = self.denom_slope

    def get_best_slope(self, cl_curve):
        """
        EXECUTION PHASE: Runs thousands of times.
        Vectorized calculation of 3 slopes and 3 R2s.
        """
        # 1. Extract Cl segments (Fancy indexing)
        # We assume cl_curve matches alpha_grid length
        # We manually stack the slices to get a (3, N) array
        # This is the only "slow" part, but faster than loops
        ys = np.vstack(
            [
                cl_curve[self.indices[0]],
                cl_curve[self.indices[1]],
                cl_curve[self.indices[2]],
            ]
        )

        # 2. Calculate Y Statistics (Vectorized across the 3 rows)
        Sy = np.sum(ys, axis=1)
        Syy = np.sum(ys**2, axis=1)
        Sxy = np.sum(self.alphas * ys, axis=1)  # Element-wise mult then sum

        # 3. Calculate Numerator (common to Slope and R2)
        # Num = n*Sxy - Sx*Sy
        numerator = self.n * Sxy - self.Sx * Sy

        # 4. Calculate R2 for decision making
        # R2 = Num^2 / ( (n*Sxx - Sx^2) * (n*Syy - Sy^2) )
        denom_y = self.n * Syy - Sy**2

        # Avoid division by zero
        denom_total = self.denom_r2_part1 * denom_y

        # We calculate R2 scores
        # (We use a safe divide or just ignore 0 denoms)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2_scores = (numerator**2) / denom_total
            r2_scores = np.nan_to_num(r2_scores)  # Handle NaNs

        # 5. Pick the winner
        best_idx = np.argmax(r2_scores)

        # 6. Calculate Slope of the winner
        # m = Num / Denom_Slope
        best_slope = numerator[best_idx] / self.denom_slope[best_idx]

        return best_slope


def parameter_grid_generator(bounds, points_per_param):
    # --- 2. GENERATE GRID ---
    # Create the individual 1D arrays for each dimension
    grids = [np.linspace(b[0], b[1], int(n)) for b, n in zip(bounds, points_per_param)]

    # Generate the full cartesian product
    # Note: 'indexing="ij"' is often preferred for matrix indexing,
    # but default works fine for simple flattening.
    mesh = np.meshgrid(*grids, indexing="ij")

    # Flatten and stack to get shape (N_total_cases, M_parameters)
    grid_coefficients = np.vstack([m.flatten() for m in mesh]).T

    return grid_coefficients
