import numpy as np
from util import generate_lhs
from evaluation import AirfoilVarEvaluator
import aerosandbox
from concurrent.futures import ProcessPoolExecutor
from airfoil import parse_airfoil_coordinates
from train_data import save_data, histogram_train_data

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

    save_data(profile_samples, results, "surrogate_train_data")

    histogram_train_data(results)
