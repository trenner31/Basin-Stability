import numpy as np
import matplotlib.pyplot as plt


def forest_cover_derivative(C_crit, x, r, C, A):
    """Compute derivative of forest cover."""
    C = np.asarray(C)
    A = np.asarray(A)
    result = np.empty_like(C)

    mask = C > C_crit
    result[~mask] = -x * C[~mask]
    result[mask] = r * (1 - C[mask]) * C[mask] - x * C[mask]

    return result


def forest_cover(C_crit, x, r, C, A):
    """Forest cover state function."""
    C = np.asarray(C)
    A = np.asarray(A)
    result = np.empty_like(C)

    mask = (1 - x / r > calc_C_crit(A)) & (C > calc_C_crit(A))
    result[~mask] = -1
    result[mask] = 1

    return result


def calc_C_crit(A):
    """Critical forest cover threshold."""
    return A * 0.8 - 0.1


# -------------------------------
# Parameters
# -------------------------------
resolution = int(1e3)
x = 0.4
r = 1.5

A_min, A_max = 0, 1.3
C_min, C_max = 0, 1.0

A = np.linspace(A_min, A_max, resolution)
C = np.linspace(C_min, C_max, resolution)

C_crit = calc_C_crit(A)

# Create mesh grid
AA, CC = np.meshgrid(A, C)
State = forest_cover(C_crit, x, r, CC, AA)

# -------------------------------
# Plot settings
# -------------------------------
figsize = (8, 6)
cmap = "coolwarm_r"
output_file = "results/test.png"

xlabel = "Aridity (A)"
ylabel = "Forest Cover (C)"
title = "Forest.Savanna Model"

# Optional: Werte für Farblegende
cbar_label = "State (-1: low cover, 1: high cover)"

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=figsize)
mesh = plt.pcolormesh(AA, CC, State, shading="auto", cmap=cmap)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
cbar = plt.colorbar(mesh)
cbar.set_label(cbar_label)

plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.show()
