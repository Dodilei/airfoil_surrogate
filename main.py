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

alpha_sim = np.concat(
    (np.linspace(-2.5, 2, 10), np.linspace(4, 10, 4), np.linspace(12, 17, 11))
)
alpha_target = np.linspace(-3, 17, 41)

evaluator = AirfoilVarEvaluator(
    base_foil_ct=(t1, yc0, yt0),
    alpha_sim=alpha_sim,
    alpha_target_specs=(-3, 17, 41),
    slope_window_center=1.0,
    slope_window_size=5.0,
    slope_window_displacement=2.5,
)

if __name__ == "__main__":
    profile_samples = generate_lhs(bounds, n_samples=100)

    with ProcessPoolExecutor() as executor:
        results = list(
            executor.map(
                lambda X: evaluator.evaluate_variation(X, xfoil_path=xfoil_path),
                profile_samples,
            )
        )

    results = np.array(results)

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
