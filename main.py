import numpy as np
import neuralfoil
import aerosandbox
from airfoil import (
    airfoil_shape,
    airfoil_coordinates,
    fit_airfoil_shape,
    parse_airfoil_coordinates,
)

# Get S1223 coordinates from aerosandbox
s1223 = aerosandbox.Airfoil("S1223")
xy_coords = np.array(s1223.coordinates).T  # Shape (2, N)

t1, yu, yl = parse_airfoil_coordinates(xy_coords)

fit_airfoil_shape((t1, yu, yl), order_camber=6, order_thick=5, visualize=True)
