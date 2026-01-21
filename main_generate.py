import numpy as np
from util import generate_lhs
from evaluation import AirfoilVarEvaluator
import aerosandbox
from concurrent.futures import ProcessPoolExecutor
from airfoil import parse_airfoil_coordinates


xfoil_path = "C:\\Portable\\xfoil.exe"

# Get S1223 coordinates from aerosandbox
s1223 = aerosandbox.Airfoil("S1223")

t1, yu, yl = parse_airfoil_coordinates(np.array(s1223.coordinates).T, refit=200)

yc0 = 0.5 * (yu + yl)
yt0 = 2 * (yu - yc0)

original_specs = np.array(
    [0.12139, 0.1972, 0.08676, 0.47621]
)  # log(R), tmax, tmaxpos, cmax, cmaxpos

perc = 0.1
bounds = original_specs[:, np.newaxis] * (1.0 + np.array([-perc, perc]))[np.newaxis, :]

log_reynolds_bounds = np.array([np.log(1e5), np.log(4e5)])

bounds = np.vstack((log_reynolds_bounds[np.newaxis, :], bounds))

alpha_sim = np.concatenate(
    (
        np.linspace(-2.5, 2, 8),
        np.linspace(4, 8, 4),
        np.linspace(9, 15, 11),
        np.linspace(16, 17, 3),
    )
)

evaluator = AirfoilVarEvaluator(
    base_foil_ct=(t1, yc0, yt0),
    alpha_sim=alpha_sim,
    alpha_target_specs=(-3, 17, 41),
    slope_window_center=4.5,
    slope_window_size=7.0,
    slope_window_displacement=2.0,
)

evaluator.xfoil_config_set(path=xfoil_path, max_iter=75, timeout=35)


def evaluator_function(X):
    return evaluator.evaluate_variation(X)


if __name__ == "__main__":
    print("Generating samples")
    profile_samples = generate_lhs(bounds, n_samples=2000)

    print("Evaluating samples (MP)...")
    print(end="")

    with ProcessPoolExecutor() as executor:
        results = list(
            executor.map(
                evaluator_function,
                profile_samples,
            )
        )

    results = np.array(results)

    print("Done.")
    print()

    print(
        f"Non-convergence rate: {np.isnan(results.sum(axis=1)).sum() / results.shape[0] * 100:.2f}%"
    )

    print()
    print("Saving results to CSV...", end="")
    import pandas as pd

    all_data = np.concat([profile_samples, results], axis=1)

    col_names = [
        "log_Re",
        "t_max",
        "x_t",
        "m_max",
        "x_c",
        "cl_max",
        "cd_at_clmax",
        "cm_at_clmax",
        "cl_at_cdmin",
        "cd_min",
        "cm_at_cdmin",
        "dclda",
    ]

    df = pd.DataFrame(all_data, columns=col_names)
    df.to_csv("surrogate_train_data.csv", index=False)

    print(" Done.")

    print()
    print("Visualizing results.")
    # Visualize histogram of results for each of the 7 specs
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    metric_names = [
        "Cl_max",
        "Cd_at_Clmax",
        "Cm_at_Clmax",
        "Cl_at_Cdmin",
        "Cd_min",
        "Cm_at_Cdmin",
        "dCl/dAlpha",
    ]
    for i in range(7):
        ax = axes.flatten()[i]
        ax.hist(results[:, i], bins=20, color="skyblue", edgecolor="black")
        ax.set_title(metric_names[i])
        ax.grid(True)
    plt.tight_layout()
    plt.show()
