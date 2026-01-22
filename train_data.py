import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    "a0L",
]

OUTPUT_PHYSICAL_BOUNDS = [
    (1.5, 2.9),  # cl_max
    (0.02, 0.09),  # cd_at_clmax
    (-0.35, -0.10),  # cm_at_clmax
    (0.5, 1.75),  # cl_at_cdmin
    (0.00, 0.04),  # cd_min
    (-0.4, -0.1),  # cm_at_cdmin
    (0.08, 0.15),  # dclda
    (-15.0, -3.0),  # a0L
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


def save_data(input_samples, results, filename="surrogate_train_data"):
    print()
    print("Saving data to CSV...", end="")

    all_data = np.concat([input_samples, results], axis=1)

    if all_data.shape[1] != len(INPUT_COLS + OUTPUT_COLS):
        raise ValueError

    df = pd.DataFrame(all_data, columns=INPUT_COLS + OUTPUT_COLS)
    df.to_csv(filename + ".csv", index=False)

    print(" Done.")


def load_data():
    df = pd.read_csv(CSV_PATH)
    print(f"--- Loaded Data: {len(df)} rows ---")
    print()

    return df


def clean_data(
    df,
    output_target=None,
    nan_filter=False,
    bounds_filter=False,
    mad_filter=False,
    mad_filter_threshold=4.0,
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
            mad_mask = get_mad_mask(
                df_clean[col].values, threshold=mad_filter_threshold
            )
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


def audit_data_cleaning(df_original, df_cleaned):
    # 1. Identify what was dropped
    dropped_indices = df_original.index.difference(df_cleaned.index)
    df_dropped = df_original.loc[dropped_indices]

    print("--- Audit Report ---")
    print(f"Total Samples: {len(df_original)}")
    print(
        f"Dropped: {len(df_dropped)} ({len(df_dropped) / len(df_original) * 100:.1f}%)"
    )

    # 2. Plot Distributions
    fig, axes = plt.subplots(1, len(INPUT_COLS), figsize=(20, 4))

    for i, col in enumerate(INPUT_COLS):
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


def histogram_train_data(data, bins=30):
    print()
    print("Visualizing results.")

    if isinstance(data, pd.DataFrame):
        results = data[OUTPUT_COLS].values
    else:
        results = data

    # Visualize histogram of results for each of the 7 specs

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i in range(len(OUTPUT_COLS)):
        ax = axes.flatten()[i]
        ax.hist(results[:, i], bins=20, color="skyblue", edgecolor="black")
        ax.set_title(OUTPUT_COLS[i])
        ax.grid(True)
    plt.tight_layout()
    plt.show()
