import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def forest_cover_derivative(C_crit, x, r, C, A):
    """Compute derivative of forest cover."""
    C = np.asarray(C)
    A = np.asarray(A)
    result = np.empty_like(C)

    mask = C > C_crit
    result[~mask] = -x * C[~mask]
    result[mask] = r * (1 - C[mask]) * C[mask] - x * C[mask]

    return result


def forest_cover(C_crit, x, r, m, b, C, A):
    """Forest cover state function."""
    C = np.asarray(C)
    A = np.asarray(A)
    result = np.empty_like(C)

    mask = (1-x/r > calc_C_crit(A,m,b)) & (C>calc_C_crit(A,m,b))
    result[~mask] = -1
    result[mask] = 1

    return result


def calc_C_crit(A,m,b):
    """Critical forest cover threshold."""
    return A*m+b

def calc_C_F(x,r):
    return(1-x/r)

def calc_A_F(x,r,m,b):
    return (calc_C_F(x,r)-b)/m

# -------------------------------
# Parameters
# -------------------------------
resolution = int(1e3)
resolution_sparse=20
x = 0.4
r = 1.5
m=0.8
b=-0.1

A_min, A_max = 0, 1.3
C_min, C_max = 0, 1.0

A = np.linspace(A_min, A_max, resolution)
C = np.linspace(C_min, C_max, resolution)

C_crit = calc_C_crit(A,m,b)

A_sparse=np.linspace(0,1.3,resolution_sparse)
C_sparse=np.linspace(0,1.0,resolution_sparse)
C_crit_sparse=calc_C_crit(A_sparse,m,b)

# Create mesh grid
AA, CC = np.meshgrid(A, C)
AA_sparse, CC_sparse = np.meshgrid(A_sparse, C_sparse)

State=forest_cover(C_crit,x,r,m,b,CC,AA)
Direction_sparse=(forest_cover_derivative(C_crit_sparse,x,r,CC_sparse,AA_sparse))
# -------------------------------
# Plot settings
# -------------------------------
figsize = (8, 6)
cmap = ListedColormap(["#008143", "#ff9900"])
output_file = "results/test.png"
output_file_arrows = "results/test_arrows.png"

xlabel = "Aridity (A)"
ylabel = "Forest Cover (C)"
title = "Forest-Savanna Model"

# Optional: Werte für Farblegende
cbar_label = "State (-1: low cover, 1: high cover)"

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=figsize)
mesh = plt.pcolormesh(AA, CC, State, shading="auto", cmap=cmap)
plt.pcolormesh(AA, CC,State, shading='auto', cmap=cmap)#, norm=norm)
plt.plot(A,C_crit,color='white')
plt.plot([0,calc_A_F(x,r,m,b)],[calc_C_F(x,r),calc_C_F(x,r)],color='white')
plt.scatter([calc_A_F(x,r,m,b)],[calc_C_F(x,r)],color='white')
plt.plot([calc_A_F(x,r,m,b),calc_A_F(x,r,m,b)],[0,1],linestyle='--',color='white')
plt.plot([0,1.3],[calc_C_F(x,r),calc_C_F(x,r)],linestyle='--',color='white')
plt.ylim(0,1)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
# cbar = plt.colorbar(mesh)
# cbar.set_label(cbar_label)
legend_handles = [
    matplotlib.patches.Patch(color=cmap.colors[0], label="Forest"),
    matplotlib.patches.Patch(color=cmap.colors[1], label="Savanna")
]
plt.legend(handles=legend_handles, title="State", loc="upper right")
plt.tight_layout()
plt.tight_layout()
plt.savefig(output_file, dpi=300)
# plt.show()

plt.quiver(AA_sparse,CC_sparse, 0*np.ones_like(AA_sparse), Direction_sparse)



plt.savefig(output_file_arrows, dpi=300)