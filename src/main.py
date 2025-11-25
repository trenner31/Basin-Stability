import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize, TwoSlopeNorm
from functools import partial
from scipy.integrate import odeint

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

def make_plot_noarrows(A,C,func_cover,func_crit,C_F,cmap=ListedColormap(['lightgray','grey']),xlabel=None,ylabel=None,title=None,figsize=None):
        
    AA, CC = np.meshgrid(A, C)
    C_crit=func_crit(A)
    A_crit=A[np.argmin(abs(C_crit-C_F))]
    State=func_cover(C_crit=C_crit,C=CC,A=AA)

    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(AA, CC,State, shading='auto', cmap=cmap)
    ax.plot(A,C_crit,color='white')
    ax.plot([0,A_crit],[C_F,C_F],color='white')
    
    ax.plot([A_crit,A_crit],[0,1],linestyle='--',color='white')
    ax.plot([0,np.max(A)],[C_F,C_F],linestyle='--',color='white')

    ax.scatter([A_crit],[C_F],color='white')
    
    xticks =  [A_crit]
    xlabels = [ r"$A_{crit}$"]
    ax.set_xticks(xticks, xlabels)
    yticks = list(ax.get_yticks()) + [C_F]
    ylabels = [*ax.get_yticklabels(), r"$C_F=$"+format(C_F,".1f")]
    ax.set_yticks(yticks, ylabels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)

    ax.set_ylim(0,1)

    legend_handles = [
        matplotlib.patches.Patch(color=cmap.colors[0], label="Savanna"),
        matplotlib.patches.Patch(color=cmap.colors[1], label="Forest")
    ]
    ax.legend(handles=legend_handles, title="State", loc="upper right")
    fig.tight_layout()
    
    return fig,ax

def make_plot_arrows(A,C,func_cover,func_cover_derivative,func_crit,C_F,resolution_sparse=20,cmap=ListedColormap(['lightgray','grey']),xlabel=None,ylabel=None,title=None,figsize=None):
        
    fig,ax=make_plot_noarrows(figsize=figsize,
                    A=A, C=C,
                    func_cover=func_cover,
                    func_crit=func_crit,
                    C_F=C_F,
                    cmap=cmap,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=title,)
    
    A_sparse=np.linspace(np.min(A),np.max(A),resolution_sparse)
    C_sparse=np.linspace(np.min(C),np.max(C),resolution_sparse)
    C_crit_sparse=func_crit(A_sparse)
    AA_sparse, CC_sparse = np.meshgrid(A_sparse, C_sparse)
    Direction_sparse=func_cover_derivative(C_crit=C_crit_sparse,C=CC_sparse,A=AA_sparse)
    ax.quiver(AA_sparse,CC_sparse, 0*np.ones_like(AA_sparse), Direction_sparse)

    return fig,ax

def make_plot_potential(A_range,C,func_crit,func_potential,A_crit,C_F,colors,xlabel=None,ylabel=None,title=None,figsize=None,):

    fig, ax = plt.subplots(figsize=figsize)

    for A_int,i,col in zip(A_range,range(len(A_range)),colors):
        C_crit_int=func_crit(A_int)
        ax.plot(C,func_potential(C_crit=C_crit_int,C=C)+i*3e-3, label=r'A='+format(A_int/A_crit, ".1f")+r'$A_{crit}$', color=col)
        ax.scatter(C_crit_int,func_potential(C_crit=C_crit_int,C=C_crit_int)+i*3e-3, color=col)
    ax.axvline(C_F, linestyle='--', color='black')
    ax.axvline(0, linestyle='--',color='black')

    xticks = list(ax.get_xticks()) + [0.0, C_F]
    xlabels = [*ax.get_xticklabels(), r"$C_S=$"+format(0.0,".1f"), r"$C_F=$"+format(C_F,".1f")]
 
    ax.set_xticks(xticks, xlabels, rotation=35)
    ax.set_xlim(-3e-2,1)

    circle_marker = matplotlib.lines.Line2D([], [], marker='o',color='black',markersize=6,linestyle='None',label=r'$C_{crit}(A)$')
    handles,_x = ax.get_legend_handles_labels()
    handles.append(circle_marker)
    ax.legend(handles=handles, loc='upper left')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()

    return fig,ax

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

A_range=np.linspace(0,calc_A_crit(x,r,m,b)*1.2,7)

# -------------------------------
# Plot settings
# -------------------------------
figsize = (8, 6)
cmap = ListedColormap(["#ff9900","#006635"])
colors=[
    "#115a00",
    "#26a200",
    "#0fb300",
    "#44ff00",
    "#eaff00",
    "#ff9900",
    "#bc2d05",
    ]
colors2=[
    "#115a00",
    "#eaff00",
    "#bc2d05",
    ]
cmap2 = LinearSegmentedColormap.from_list("green_yellow_red", colors2).reversed()

output_file = "results/test.png"
output_file_arrows = "results/test_arrows.png"
output_file_potential = "results/test_potential.png"

xlabel = "Aridity (A)"
ylabel = "Forest Cover (C)"
title = "Forest-Savanna Model"

# -------------------------------
# Plotting
# -------------------------------
fig,_=make_plot_noarrows(A=A, C=C,
                         func_cover=partial(forest_cover,x=x,r=r,m=m,b=b),
                         func_crit=partial(calc_C_crit,m=m,b=b),
                         C_F=calc_C_F(x=x,r=r),
                         cmap=cmap,
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=title,
                         figsize=figsize,
                        )
fig.savefig(output_file, dpi=300)
plt.close()

fig,_=make_plot_arrows(A=A, C=C,
                       func_cover=partial(forest_cover,x=x,r=r,m=m,b=b),
                       func_cover_derivative=partial(forest_cover_derivative, x=x, r=r),
                       func_crit=partial(calc_C_crit,m=m,b=b),
                       C_F=calc_C_F(x=x,r=r),
                       cmap=cmap,
                       xlabel=xlabel,
                       ylabel=ylabel,
                       title=title,
                       figsize=figsize,
                       )
fig.savefig(output_file_arrows, dpi=300)
plt.close()

fig,_ = make_plot_potential(A_range=A_range,
                            C=C,
                            colors=colors,
                            func_crit=partial(calc_C_crit,m=m,b=b),
                            func_potential=partial(calc_U,x=x,r=r),
                            A_crit=calc_A_crit(x=x,r=r,m=m,b=b),
                            C_F=calc_C_F(x=x,r=r),
                            xlabel='Forest Cover (C)',
                            ylabel='Potential (U)',
                            title=title,
                            figsize=figsize,
                            )
fig.savefig(output_file_potential, dpi=300)
plt.close()




# ##### TODO SAUBER

# def make_plot_trajectories(A_range,figsize=None):
#     fig,axs = plt.subplots(figsize)
#     fig.tight_layout()
#     return fig,axs

# critical = 0.5 
# norm = TwoSlopeNorm(
#     vmin=0.0,
#     vcenter=0.3,
#     vmax=1.0
# )
# def dcdt(C,t,A):
#     return forest_cover_derivative(C_crit=calc_C_crit(A=A,m=m,b=b),x=x,r=r,C=C,A=A)

# A_range=np.linspace(A_min,A_max,5)
# fig,axs=plt.subplots(len(A_range), figsize=(6,12), sharex=True)
# for A,ax in zip(A_range,axs):
#     dc=partial(dcdt,A=A)
#     t = np.linspace(0, 7, 200)
#     for C0 in np.linspace(0,1,10):
#         C_solved = odeint(dc, C0, t)
#         ax.plot(t,C_solved,label=str(C0)+str(forest_cover(C_crit=calc_C_crit(A=A,m=m,b=b),x=x,r=r,m=m,b=b,C=C0,A=A)),color=cmap(norm(C0)))
#     ax.set_ylabel('C(t)')
# axs[-1].set_xlabel('t')

# fig.savefig('results/tests.png')
