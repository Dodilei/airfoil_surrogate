import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from data_manager import save_surrogate_pack

# ==========================================
# 1. CONFIGURATION
# ==========================================
CSV_PATH = "surrogate_train_data.csv"

# Column Mapping (Adjust these to match your exact CSV headers)
INPUT_COLS = ["log_Re", "t_max", "x_t", "c_max", "x_c"]
OUTPUT_COLS = [
    "cl_max",
    "cd_at_clmax",
    "cm_at_clmax",
    "cl_at_cdmin",
    "cd_min",
    "cm_at_cdmin",
    "dclda",
]

OUTPUT_PHYSICAL_BOUNDS = [
    (1.5, 2.9),  # cl_max
    (0.02, 0.09),  # cd_at_clmax
    (-0.35, -0.10),  # cm_at_clmax
    (0.5, 1.75),  # cl_at_cdmin
    (0.00, 0.04),  # cd_min
    (-0.4, -0.1),  # cm_at_cdmin
    (0.08, 0.15),  # dclda
]


def get_mad_mask(points, threshold=4.0):
    """
    Returns a boolean mask: True for valid points, False for outliers.
    Threshold 3.5 is the standard recommendation by Iglewicz and Hoaglin.
    """
    if len(points.shape) > 1:
        points = points.ravel()  # Ensure it's a 1D array

    median = np.median(points)
    ad = np.abs(points - median)  # Absolute Deviations
    mad = np.median(ad)  # Median Absolute Deviation

    # Avoid division by zero if all points are identical
    if mad == 0:
        return np.ones_like(points, dtype=bool)

    modified_z_scores = 0.6745 * (points - median) / mad

    return np.abs(modified_z_scores) <= threshold


# ==========================================
# 2. DATA LOADING & CLEANING
# ==========================================
def load_data():
    df = pd.read_csv(CSV_PATH)
    print(f"--- Loaded Data: {len(df)} rows ---")
    print()

    return df


def clean_data(
    df, output_target=None, nan_filter=True, bounds_filter=True, mad_filter=True
):
    df_clean = df.copy()

    # A. Sanity Filter: Remove failed runs (NaNs)
    initial_count = len(df)
    if nan_filter:
        df_clean.dropna(subset=INPUT_COLS + OUTPUT_COLS, inplace=True)

        step1_count = len(df_clean)
        print("Removed NaN rows")
        print(f"Valid data rate: {step1_count / initial_count}")
    else:
        step1_count = initial_count

    print()
    for col in OUTPUT_COLS:
        if output_target and col != output_target:
            continue
        print(
            f"Bounds for {col}: [{df_clean[col].min():.3f}, {df_clean[col].max():.3f}]"
        )

    # B. Physics Filter
    if bounds_filter:
        print()
        print("Applying physical bounds filter:")
        outlier_masks = []
        for bounds, col in zip(OUTPUT_PHYSICAL_BOUNDS, OUTPUT_COLS):
            if output_target and col != output_target:
                continue
            outlier_mask = (df_clean[col] < bounds[0]) | (df_clean[col] > bounds[1])
            outlier_masks.append(outlier_mask)
            print(
                f"    Found {outlier_mask.sum()} outliers ({(outlier_mask.sum() / len(df_clean)):.2f}%) from {col} based on physical bounds."
            )

        total_outliers = np.any(np.array(outlier_masks), axis=0)
        df_clean = df_clean[~total_outliers]
        step2_count = len(df_clean)
        print(
            f"Removed {step1_count - step2_count} outliers ({((step1_count - step2_count) / step1_count):.2f}%)"
        )
    else:
        step2_count = step1_count

    # C. MAD Filter
    if mad_filter:
        print()
        print("Applying MAD outlier filter:")
        mad_outlier_masks = []
        for col in OUTPUT_COLS:
            if output_target and col != output_target:
                continue
            mad_mask = get_mad_mask(df_clean[col].values)
            mad_outlier_masks.append(~mad_mask)
            mad_outliers = df_clean[~mad_mask].values
            try:
                bounds = (mad_outliers.min(), mad_outliers.max())
            except ValueError:
                bounds = (np.nan, np.nan)
            print(
                f"    Found {np.sum(~mad_mask)} outliers ({(np.sum(~mad_mask) / len(df_clean)):.2f}%) from {col} using MAD. Bounds: [{bounds[0]:.3f}, {bounds[1]:.3f}]"
            )

        total_mad_outliers = np.any(np.array(mad_outlier_masks), axis=0)
        df_clean = df_clean[~total_mad_outliers]
        step3_count = len(df_clean)
        print(
            f"Removed {step2_count - step3_count} outliers ({((step2_count - step3_count) / step2_count):.2f}%) using MAD."
        )
    else:
        step3_count = step2_count

    print()
    print("--- Data Cleaning ---")
    print(f"Original: {initial_count}")
    print(
        f"Cleaned:  {len(df_clean)} (Removed {initial_count - len(df_clean)} outliers)"
    )

    return df_clean


def audit_data_cleaning(df_original, df_cleaned, input_params):
    # 1. Identify what was dropped
    dropped_indices = df_original.index.difference(df_cleaned.index)
    df_dropped = df_original.loc[dropped_indices]

    print("--- Audit Report ---")
    print(f"Total Samples: {len(df_original)}")
    print(
        f"Dropped: {len(df_dropped)} ({len(df_dropped) / len(df_original) * 100:.1f}%)"
    )

    # 2. Plot Distributions
    fig, axes = plt.subplots(1, len(input_params), figsize=(20, 4))

    for i, col in enumerate(input_params):
        # Plot Kept Data
        sns.kdeplot(df_cleaned[col], ax=axes[i], label="Kept", fill=True, color="green")
        # Plot Dropped Data
        sns.kdeplot(
            df_dropped[col], ax=axes[i], label="Dropped", fill=True, color="red"
        )

        axes[i].set_title(f"Distribution: {col}")
        axes[i].legend()

    plt.tight_layout()
    plt.show()


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


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load
    df = load_data()
    df_clean = clean_data(df)

    # Optional: Audit Cleaning
    # audit_data_cleaning(df, df_clean, INPUT_COLS)

    df_sample = df_clean.sample(n=100, random_state=42)

    models = {}
    scalers = {}

    # 2. Train
    for output_target in OUTPUT_COLS:
        print()
        print(f"--- Surrogate for {output_target} ---")
        gp_model, scaler, X_test, y_test, y_pred, y_std, r2, rmse = train_surrogate(
            df_sample, output_target=output_target
        )

        models[output_target] = gp_model
        scalers[output_target] = scaler

        # 3. Validate
        # plot_validation(gp_model, scaler, X_test, y_test, y_pred, y_std, df)

    save_surrogate_pack(models, scalers)
