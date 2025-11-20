import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from functools import partial

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
    return np.maximum(A*m+b,0)

def calc_C_F(x,r):
    return(1-x/r)

def calc_A_crit(x,r,m,b):
    return (calc_C_F(x,r)-b)/m

def calc_U(x,r,C_crit,C):
    return x/2*C**2+ (C>C_crit)*(r/3 * (C**3-C_crit**3) - r/2 *(C**2-C_crit**2))

def make_plot_noarrows(figsize,A,C,func_cover,func_crit,cmap):
            # detect global variable use
        import inspect
        globals_used = set(make_plot_noarrows.__code__.co_names) - set(locals())
        print(globals_used)
        AA, CC = np.meshgrid(A, C)
        C_crit=func_crit(A)
        State=func_cover(C_crit=C_crit,C=CC,A=AA)
        fig, ax = plt.subplots(figsize=figsize)
        ax.pcolormesh(AA, CC,State, shading='auto', cmap=cmap)
        plt.plot(A,C_crit,color='white')
        plt.plot([0,calc_A_crit(x,r,m,b)],[calc_C_F(x,r),calc_C_F(x,r)],color='white')
        # plt.scatter([calc_A_crit(x,r,m,b)],[calc_C_F(x,r)],color='white')
        # plt.plot([calc_A_crit(x,r,m,b),calc_A_crit(x,r,m,b)],[0,1],linestyle='--',color='white')
        # plt.plot([0,1.3],[calc_C_F(x,r),calc_C_F(x,r)],linestyle='--',color='white')
        # plt.ylim(0,1)
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt.title(title)
        # ticks =  [calc_A_crit(x,r,m,b)]
        # labels = [ r"$A_{crit}$"]
        # plt.xticks(ticks, labels)
        # ticks = list(plt.yticks()[0]) + [calc_C_F(x,r)]
        # labels = [*plt.yticks()[1], r"$C_F=$"+format(calc_C_F(x,r),".1f")]
        # plt.yticks(ticks, labels)
        # # cbar = plt.colorbar(mesh)
        # # cbar.set_label(cbar_label)
        # legend_handles = [
        #     matplotlib.patches.Patch(color=cmap.colors[0], label="Savanna"),
        #     matplotlib.patches.Patch(color=cmap.colors[1], label="Forest")
        # ]
        # plt.legend(handles=legend_handles, title="State", loc="upper right")
        plt.tight_layout()
        return fig
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
# AA, CC = np.meshgrid(A, C)
# AA_sparse, CC_sparse = np.meshgrid(A_sparse, C_sparse)

# State=forest_cover(C_crit,x,r,m,b,CC,AA)
# Direction_sparse=(forest_cover_derivative(C_crit_sparse,x,r,CC_sparse,AA_sparse))
# -------------------------------
# Plot settings
# -------------------------------
figsize = (8, 6)
cmap = ListedColormap(["#ff9900","#006635"])
output_file = "results/test.png"
output_file_arrows = "results/test_arrows.png"
output_file_potential = "results/test_potential.png"

xlabel = "Aridity (A)"
ylabel = "Forest Cover (C)"
title = "Forest-Savanna Model"

# # Optional: Werte für Farblegende
# cbar_label = "State (-1: low cover, 1: high cover)"

# -------------------------------
# Plot
# -------------------------------
# plt.figure(figsize=figsize)
# mesh = plt.pcolormesh(AA, CC, State, shading="auto", cmap=cmap)
# plt.pcolormesh(AA, CC,State, shading='auto', cmap=cmap)#, norm=norm)
# plt.plot(A,C_crit,color='white')
# plt.plot([0,calc_A_crit(x,r,m,b)],[calc_C_F(x,r),calc_C_F(x,r)],color='white')
# plt.scatter([calc_A_crit(x,r,m,b)],[calc_C_F(x,r)],color='white')
# plt.plot([calc_A_crit(x,r,m,b),calc_A_crit(x,r,m,b)],[0,1],linestyle='--',color='white')
# plt.plot([0,1.3],[calc_C_F(x,r),calc_C_F(x,r)],linestyle='--',color='white')
# plt.ylim(0,1)
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.title(title)
# ticks =  [calc_A_crit(x,r,m,b)]
# labels = [ r"$A_{crit}$"]
# plt.xticks(ticks, labels)
# ticks = list(plt.yticks()[0]) + [calc_C_F(x,r)]
# labels = [*plt.yticks()[1], r"$C_F=$"+format(calc_C_F(x,r),".1f")]
# plt.yticks(ticks, labels)
# # cbar = plt.colorbar(mesh)
# # cbar.set_label(cbar_label)
# legend_handles = [
#     matplotlib.patches.Patch(color=cmap.colors[0], label="Savanna"),
#     matplotlib.patches.Patch(color=cmap.colors[1], label="Forest")
# ]
# plt.legend(handles=legend_handles, title="State", loc="upper right")
# plt.tight_layout()
fig=make_plot_noarrows(figsize=figsize,
                       A=A, C=C,
                       func_cover=partial(forest_cover,x=x,r=r,m=m,b=b),
                       func_crit=partial(calc_C_crit,m=m,b=b),
                       func_stable=partial(calc_C_F,x=x,r=r)
                       cmap=cmap,)
fig.savefig(output_file, dpi=300)
# plt.show()

# plt.quiver(AA_sparse,CC_sparse, 0*np.ones_like(AA_sparse), Direction_sparse)

# plt.tight_layout(,
# plt.savefig(output_file_arrows, dpi=300)

# plt.cla()
# A_range=np.linspace(0,calc_A_crit(x,r,m,b)*1.2,7)
# colors=[
#     "#bc2d05",
#     "#ff9900",
#     "#eaff00",
#     "#44ff00",
#     "#0fb300",
#     "#26a200",
#     "#115a00",
#     ]
# for A_int,i,col in zip(A_range,range(7),reversed(colors)):
#     C_crit_int=calc_C_crit(A_int,m,b)
#     plt.plot(C,calc_U(x,r,C_crit_int,C)+i*3e-3, label=r'A='+format(A_int/calc_A_crit(x,r,m,b), ".1f")+r'$A_{crit}$', color=col)
#     plt.scatter(C_crit_int,calc_U(x,r,C_crit_int,C_crit_int)+i*3e-3, color=col)
# plt.plot([calc_C_F(x,r),calc_C_F(x,r)],[-0.15,0.25], linestyle='--', color='black')
# plt.plot([0,0],[-0.15,0.25], linestyle='--',color='black')

# ticks = list(plt.xticks()[0]) + [0.0, calc_C_F(x,r)]
# labels = [*plt.xticks()[1], r"$C_S=$"+format(0.0,".1f"), r"$C_F=$"+format(calc_C_F(x,r),".1f")]
# plt.xticks(ticks, labels)
# plt.xlim(-0.05,1)
# plt.ylim(-0.15,0.25)
# plt.scatter(10,10,color='black', label=r'$C_{crit}(A)$')
# plt.legend(loc='upper left')
# plt.xlabel('C')
# plt.ylabel('Potential')
# plt.title(title)
# plt.tight_layout()
# plt.savefig(output_file_potential, dpi=300, bbox_inches='tight')