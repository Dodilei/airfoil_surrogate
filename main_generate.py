import numpy as np
import aerosandbox

import warnings

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from util import generate_lhs
from evaluation import AirfoilVarEvaluator
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
)  # tmax, tmaxpos, cmax, cmaxpos

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
)

evaluator.xfoil_config_set(path=xfoil_path, max_iter=75, timeout=35)


def evaluator_function(X):
    return evaluator.evaluate_variation(X)


if __name__ == "__main__":
    n_cores = 7

    print("Generating samples")
    profile_samples = generate_lhs(bounds, n_samples=5)

    print(f"Evaluating samples (multiprocess, {n_cores} cores)...")
    print(end="")

    warnings.filterwarnings("ignore", category=UserWarning, module="aerosandbox")

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # 1. Submit all tasks and create a list of future objects
        futures = [executor.submit(evaluator_function, x) for x in profile_samples]

        # 2. Wrap as_completed in tqdm for a live progress bar
        results = []
        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Executing XFoil & Spec Evaluation",
        ):
            results.append(f.result())

    results = np.array(results)

    print("Done.")
    print()

    print(
        f"Non-convergence rate: {np.isnan(results.sum(axis=1)).sum() / results.shape[0] * 100:.2f}%"
    )

    save_data(profile_samples, results, "surrogate_train_data")

    histogram_train_data(results)
