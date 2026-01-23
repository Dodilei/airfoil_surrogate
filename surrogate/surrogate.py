import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from data.train_data import INPUT_COLS


# ==========================================
# 3. SURROGATE TRAINING (Kriging)
# ==========================================
def train_surrogate(
    df,
    output_target,
    constant_kernel_param=1.0,
    matern_nu=2.5,
    white_kernel_noise=6e-3,
    length_scale_min=1e-5,
):
    X = df[INPUT_COLS].values
    y = df[output_target].values

    # 1. Scale Inputs (Crucial for distance-based kernels like GP)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Split Data (80% Train, 20% Validate)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 3. Define Kernel
    # ConstantKernel: Adjusts magnitude
    # Matern(nu=2.5): Allows for physical smoothness (differentiable twice)
    # WhiteKernel: Handles noise in the XFoil data (prevents overfitting to spikes)
    kernel = C(constant_kernel_param) * Matern(
        length_scale=np.ones(5),
        length_scale_bounds=(length_scale_min, 1e5),
        nu=matern_nu,
    ) + WhiteKernel(noise_level=white_kernel_noise)

    print()
    print("--- Training Gaussian Process ---")
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True
    )
    gp.fit(X_train, y_train)

    # 4. Score
    y_pred, y_std = gp.predict(X_test, return_std=True)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Model R^2: {r2:.4f}")
    print(f"Model RMSE: {rmse:.4f}")

    return gp, scaler, X_test, y_test, y_pred, y_std, r2, rmse


def evaluate_surrogate_physics(
    model,
    scaler,
    df,
    output_name,
    physics_slice_input=["c_max"],
    slice_resolution=200,
    plot=False,
):
    """
    Evaluates model accuracy on random data AND checks physics consistency
    on a synthetic slice.

    Parameters:
    - model: The trained estimator.
    - X: The unordered feature matrix (Test set).
    - y: The unordered target matrix (Test set).
    - slice_feature_idx: Index of the input feature to sweep for the physics check
      (e.g., Angle of Attack index or Main Geometry Parameter index).
    - slice_res: Number of points for the synthetic physics curve.
    - target_names: Names of the output targets (e.g., ['Max Camber', 'log(R)']).
    """

    X_arr = scaler.transform(df[INPUT_COLS].values)
    y_arr = df[output_name].values

    # --- 2. ACCURACY (Using the provided random test set) ---
    # We predict on the actual test set to get R2, RMSE, etc.
    y_pred = model.predict(X_arr)

    # --- 3. PHYSICS SLICE GENERATION (Synthetic) ---
    # To test wigglyness, we create a fake dataset that is perfectly smooth.
    # We fix all variables to their mean, and sweep ONE variable from min to max.

    X_slice = np.zeros((slice_resolution, X_arr.shape[1]))

    # Set all features to their mean value (baseline airfoil/condition)
    mean_values = np.mean(X_arr, axis=0)
    X_slice[:] = mean_values

    # A. Accuracy Metrics (Real Data)
    r2 = r2_score(y_arr, y_pred)
    rmse = np.sqrt(mean_squared_error(y_arr, y_pred))

    epsilon = 1e-10  # Prevent div by zero
    # MPE (Mean Percentage Error)
    mpe = np.mean(np.abs((y_arr - y_pred) / (y_arr + epsilon))) * 100

    print(f"{'R² (Accuracy)':<25} : {r2:.4f}")
    print(f"{'RMSE':<25} : {rmse:.4f}")
    print(f"{'Mean % Error':<25} : {mpe:.2f}%")

    if plot:
        # 1. Create the figure
        fig, axes = plt.subplots(figsize=(10, 8))

        # 2. Define a 2x2 grid layout
        gs = fig.add_gridspec(2, len(physics_slice_input))

        # 3. Assign subplots to the grid
        # First row, spanning all columns
        ax1 = fig.add_subplot(gs[0, :])

        ax1.scatter(
            y_arr,
            y_pred,
            alpha=0.5,
        )
        ax1.plot([y_arr.min(), y_arr.max()], [y_arr.min(), y_arr.max()], "r--", lw=2)
        ax1.set_xlabel("Actual XFoil Value")
        ax1.set_ylabel("Surrogate Prediction")
        ax1.set_title(f"Accuracy Plot (R2={r2_score(y_arr, y_pred):.3f})")
        ax1.grid(True)

    print("-" * 45)
    print(f"{'Metric':<25} | {'Value':<15}")

    # --- 4. CALCULATION LOOP ---
    for i, name in enumerate(physics_slice_input):
        print(f"\n### Target: {name}")
        slice_input_idx = INPUT_COLS.index(name)

        # Sweep the chosen feature (slice_feature_idx) from its min to max observed in X
        feat_min = np.min(X_arr[:, slice_input_idx])
        feat_max = np.max(X_arr[:, slice_input_idx])
        sweep_vector = np.linspace(feat_min, feat_max, slice_resolution)
        X_slice[:, slice_input_idx] = sweep_vector

        # Predict on this smooth slice
        y_slice_pred, y_slice_std = model.predict(X_slice, return_std=True)

        # B. Wigglyness Factor (Synthetic Slice Data)
        # We analyze the curve generated by y_slice_pred[:, i]

        # First Derivative (Slope)
        dy = np.gradient(y_slice_pred, sweep_vector)

        # Second Derivative (Curvature / Acceleration)
        # This captures changes in slope magnitude (e.g., getting steeper without reversing)
        d2y = np.gradient(dy, sweep_vector)

        # Wigglyness Metric: RMS of the 2nd Derivative
        # We normalize by the range of y to make it comparable across different scales
        y_range = np.max(y_slice_pred) - np.min(y_slice_pred)
        if y_range == 0:
            y_range = 1.0

        wigglyness = np.sqrt(np.mean(d2y**2)) / y_range

        # --- Display ---

        print(f"{'Smoothness (Physics)':<25} : {1 / wigglyness:.4f}")

        if plot:
            ax_ps = fig.add_subplot(gs[1, i])

            ax_ps.plot(sweep_vector, y_slice_pred, "b-", lw=2, label="Mean Prediction")
            ax_ps.fill_between(
                sweep_vector,
                y_slice_pred - 1.96 * y_slice_std,
                y_slice_pred + 1.96 * y_slice_std,
                color="blue",
                alpha=0.2,
                label="95% Conf",
            )
            ax_ps.set_xlabel(name)
            ax_ps.set_ylabel("Predicted target")
            ax_ps.set_title("Physics Check")
            ax_ps.grid(True)
            ax_ps.legend()

    if plot:
        plt.tight_layout()
        plt.show()
