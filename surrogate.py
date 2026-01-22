import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from train_data import INPUT_COLS


# ==========================================
# 3. SURROGATE TRAINING (Kriging)
# ==========================================
def train_surrogate(
    df, output_target, constant_kernel_param=1.0, matern_nu=2.5, white_kernel_noise=6e-3
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
        length_scale=np.ones(5), nu=matern_nu
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


# ==========================================
# 4. VALIDATION
# ==========================================
def plot_validation(gp, scaler, X_test, y_test, y_pred, y_std, df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot A: Prediction Accuracy
    # Ideal model follows the red dashed line perfectly
    axes[0].errorbar(
        y_test,
        y_pred,
        yerr=1.96 * y_std,
        fmt="o",
        alpha=0.5,
        ecolor="gray",
        label="95% Conf",
    )
    axes[0].plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
    )
    axes[0].set_xlabel("Actual XFoil Value")
    axes[0].set_ylabel("Surrogate Prediction")
    axes[0].set_title(f"Accuracy Plot (R2={r2_score(y_test, y_pred):.3f})")
    axes[0].grid(True)

    # Plot B: Physics Slice (Camber Sensitivity)
    # We freeze all parameters at mean, and sweep Max Camber (index 3)
    # We expect CL_max to increase with Camber. If it wiggles or drops, the model is bad.

    # Create synthetic sweep data
    N_points = 100
    means = df[INPUT_COLS].mean().values

    # Index of "Max Camber" in INPUT_COLS is 3
    camber_idx = 3
    camber_range = np.linspace(
        df.iloc[:, camber_idx].min(), df.iloc[:, camber_idx].max(), N_points
    )

    X_sweep = np.tile(means, (N_points, 1))  # Repeat mean values
    X_sweep[:, camber_idx] = camber_range  # Overwrite camber column

    # Scale and Predict
    X_sweep_scaled = scaler.transform(X_sweep)
    y_sweep, y_sweep_std = gp.predict(X_sweep_scaled, return_std=True)

    axes[1].plot(camber_range, y_sweep, "b-", lw=2, label="Mean Prediction")
    axes[1].fill_between(
        camber_range,
        y_sweep - 1.96 * y_sweep_std,
        y_sweep + 1.96 * y_sweep_std,
        color="blue",
        alpha=0.2,
        label="95% Conf",
    )
    axes[1].set_xlabel("Max Camber (m/c)")
    axes[1].set_ylabel("Predicted target")
    axes[1].set_title("Physics Check: Camber Sensitivity")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
