import numpy as np
import neuralfoil
import aerosandbox
from geometry.airfoil import (
    fit_airfoil_shape_kulfan,
    parse_airfoil_coordinates,
)

# Get S1223 coordinates from aerosandbox
s1223 = aerosandbox.Airfoil("S1223")

t1, yu, yl = parse_airfoil_coordinates(np.array(s1223.coordinates).T, refit=200)

a0_shared, A_upper, A_lower, te_thick = fit_airfoil_shape_kulfan(
    (t1, yu, yl), order=8, visualize=False
)

# Bounds for coefficients
perc = 0.2
bounds = np.array([(1 - perc, 1 + perc)] * (17))  # 8 for camber, 8 for thickness
bounds *= np.hstack((a0_shared, A_upper, A_lower))[:, np.newaxis]


def parameter_grid_generator(grids, block_size=1e6):
    """
    Yields blocks of the Cartesian product of input grids without
    allocating the full array in memory.
    """
    # 1. Calculate dimensions and total size
    block_size = int(block_size)
    grid_shapes = [len(g) for g in grids]
    total_combinations = np.prod(grid_shapes)

    # 2. Iterate through the total space in chunks
    for start_idx in range(0, total_combinations, block_size):
        end_idx = min(start_idx + block_size, total_combinations)

        # Create linear indices for the current block
        # Shape: (current_block_size,)
        linear_indices = np.arange(start_idx, end_idx)

        # 3. Convert linear indices to coordinate indices for each dimension
        # We use order='F' (Fortran-style) to mimic the default behavior
        # of np.meshgrid flattening if you were to use indexing='ij'.
        # If you strictly want the order of itertools.product, use order='C'.
        coordinate_indices = np.unravel_index(linear_indices, grid_shapes, order="F")

        # 4. Map indices back to actual values
        # This builds the block (N_block, M_params)
        block = np.column_stack(
            [grids[dim][idx] for dim, idx in enumerate(coordinate_indices)]
        )

        yield block


# --- Setup your grids ---
grid_points_per_dim = 3
# Define your base grids from bounds
grids = [
    np.linspace(bounds[i, 0], bounds[i, 1], grid_points_per_dim)
    for i in range(bounds.shape[0])
]
# Add the extra dimension you had in your example
grids.append(np.linspace(12, 15, 5))

# --- Usage ---
block_size = 2.5e6  # Adjust based on your RAM

# 'blocks' is now a generator. It calculates data only when you loop over it.
block_gen = parameter_grid_generator(grids, block_size)

CL = []
CD = []

for i, block in enumerate(block_gen):
    print(
        f"Processing block {i + 1} of {1 + (3**17 * 5) // block_size} with {block.shape[0]} samples.",
        end="",
    )

    result = neuralfoil.get_aero_from_kulfan_parameters(
        {
            "leading_edge_weight": block[:, 0],
            "upper_weights": block[:, 1:9].T,
            "lower_weights": block[:, 9:-1].T,
            "TE_thickness": 0,
        },
        alpha=block[:, -1],
        Re=480e3,
        model_size="xxsmall",
    )

    print(" Done.")

    CL.extend(result["CL"])
    CD.extend(result["CD"])

print("Finished processing all blocks.")

print("Find maximum Cl for each variation")
results = np.concatenate(
    [
        np.array(CL)[:, np.newaxis],
        np.array(CD)[:, np.newaxis],
    ],
    axis=1,
)

results = results.reshape(-1, 10, 2)
idx_clmax = results[..., 0].argmax(axis=1)
clmax_results = results[np.arange(len(idx_clmax)), idx_clmax, :]

# make a kde + scatter plot of Cl vs Cd using seaborn (sample)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

all_results = pd.concat(
    [pd.DataFrame({"Cl": clmax_results[:, 0], "Cd": clmax_results[:, 1]})],
    ignore_index=True,
)
sampled_results = all_results.sample(
    n=min(10000, all_results.shape[0]), random_state=42
)
sns.jointplot(
    data=sampled_results,
    x="Cd",
    y="Cl",
    kind="kde",
    fill=True,
    cmap="viridis",
    height=8,
)
# Style
plt.suptitle("Aerodynamic Coefficient Distribution for S1223 Variants", fontsize=16)
plt.xlabel("Drag Coefficient (Cd)", fontsize=14)
plt.ylabel("Lift Coefficient (Cl)", fontsize=14)

# Plot a point at the original S1223 values
original_result = neuralfoil.get_aero_from_kulfan_parameters(
    {
        "leading_edge_weight": a0_shared,
        "upper_weights": A_upper,
        "lower_weights": A_lower,
        "TE_thickness": 0,
    },
    alpha=np.linspace(10, 17, 10),
    Re=480e3,
    model_size="xxsmall",
)


plt.show()
