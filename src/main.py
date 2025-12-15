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

def calc_linear_stability(x,r,C_crit,C):
    """dF(C)/dC derivative of dC/dt with C_F=1-x/r, result is eigenvalue, because of r>x>0 is eigenvalue<0 ->stability, always comes back to initial state, result is how fast it goes back
    use positive value for x-r ->abs(x-r)"""
    return(C>C_crit)*(abs(x-r))

def calc_basin_stability(A,A_crit,C_crit):
    """basin=stability volume c_crit to 1 (100% coverage), A<A_crit because basin exists only if forest is stable"""
    return(A<A_crit)*(1-C_crit)
    
def dcdt(C,t,A):
    return forest_cover_derivative(C_crit=calc_C_crit(A=A,m=m,b=b),x=x,r=r,C=C,A=A)

def make_plot_noarrows(A,C,func_cover,func_crit,C_F,cmap=ListedColormap(['lightgray','grey']),xlabel=None,ylabel=None,title=None,figsize=None):
        
    AA, CC = np.meshgrid(A, C)
    C_crit=func_crit(A)
    # Berechne den Tipping Point Index
    idx_crit = np.argmin(abs(C_crit-C_F))
    A_crit = A[idx_crit]
    
    State=func_cover(C_crit=C_crit,C=CC,A=AA)

    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(AA, CC,State, shading='auto', cmap=cmap)
    
    # 1. Die Linie Plotten (ohne Label für Legende)
    ax.plot(A, C_crit, color='white', linewidth=2)
    
    # 2. Text DIREKT auf die Linie schreiben
    # Wir suchen einen Punkt ca. in der Mitte zwischen 0 und A_crit
    text_idx = int(idx_crit * 0.5) 
    text_x = A[text_idx]
    text_y = C_crit[text_idx]
    
    # Text einfügen: rotation passt den Textwinkel an die Linie an (ggf. anpassen)
    ax.text(text_x, text_y + 0.02, r'$C_{crit}(A)$', 
            color='white', 
            fontsize=12, 
            fontweight='bold', 
            rotation=28, # Winkel anpassen, falls es bei dir zu steil/flach ist
            ha='center', 
            va='bottom')

    # Hilfslinien
    ax.plot([0,A_crit],[C_F,C_F],color='white')
    ax.plot([A_crit,A_crit],[0,1],linestyle='--',color='white')
    ax.plot([0,np.max(A)],[C_F,C_F],linestyle='--',color='white')

    ax.scatter([A_crit],[C_F],color='white', zorder=5)
    
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

    # Legende nur noch für die Zustände (Farben)
    legend_handles = [
        matplotlib.patches.Patch(color=cmap.colors[0], label="Savanna State"),
        matplotlib.patches.Patch(color=cmap.colors[1], label="Forest State")
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.8)
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
    xlabels = [*ax.get_xticklabels(), r"$C_S=$"+format(0.0,".2f"), r"$C_F=$"+format(C_F,".2f")]
 
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

def make_plot_basin_vs_linear_stability(A, basin_stability, linear_stability, A_crit):
    """comparison of basin and linear stability with 2 y-axes"""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Axis 1 (left): Basin Stability (green)
    color = 'tab:green'
    ax1.set_xlabel('Aridity (A)',)
    ax1.set_ylabel('Basin Stability (Safety Volume)')
    ax1.plot(A, basin_stability, color=color, label='Basin Stability')
    # ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.05, 1.05)

    # Axis 2 (right): Linear Stability (blue)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Linear Stability (Recovery Speed)')
    ax2.plot(A, linear_stability, color=color, linestyle='--', label='Linear Stability')
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_ylim(-0.05, 1.05) 
    max_val = np.max(linear_stability)
    top_limit = max(1.05, max_val * 1.1) 
    ax2.set_ylim(-0.05, top_limit)

    # Markierungen
    ax1.axvline(x=A_crit, color='red', linestyle=':', linewidth=2, label=r'$A_{crit}$ (Tipping Point)')
    # ax1.text(A_crit, -0.05, r'$A_{crit}$ (Tipping Point)', color='red', ha='center', va='top', 
    #      transform=ax1.get_xaxis_transform(), fontsize=12)

    # # --- DYNAMISCHE TEXT-PLATZIERUNG ---
    # # 1. Linear Stability Text (Blau)
    # y_linear = linear_stability[0]
    # ax2.text(A[int(len(A)*0.3)], y_linear + 0.05, 
    #          'Recovery Speed constant', 
    #          color='tab:blue', fontweight='bold', fontsize=12, ha='left')

    # # 2. Basin Stability Text (Grün) - DEUTLICH TIEFER
    # # Wir nehmen einen Punkt ziemlich am Anfang (15% des Weges)
    # idx_pos = np.argmin(np.abs(A - (A_crit * 0.15))) 
    # x_basin = A[idx_pos]
    # y_basin = basin_stability[idx_pos]
    
    # # Das schiebt den Text weiter in den "freien Raum" unter der Kurve
    # ax1.text(x_basin, y_basin - 0.4, 
    #          'Safety Volume decreases!', 
    #          color='tab:green', fontweight='bold', fontsize=12, 
    #          ha='left', va='top') 

    # Titel und Layout
    plt.title('Linear vs. Basin Stability', fontsize=16)
    # ax1.grid(True, linestyle='--', alpha=0.5)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,)
    fig.tight_layout()
    
    return fig, (ax1, ax2)

def make_plot_trajectories(A_range,C0_range,func_C_crit,C_F,t=np.linspace(0,10,5),figsize=None):
    fig,axs = plt.subplots(len(A_range),figsize=figsize, sharex=True)
    for A,ax in zip(A_range, axs):
        dcdt_fixedA = partial(dcdt,A=A)
        center = func_C_crit(A)*(func_C_crit(A)<C_F)+(func_C_crit(A)>C_F)
        norm = normation(vmin=0.0,vcenter=center,vmax=1.0)
        cmap3 = cmap2(vmin=0.0,vcenter=center,vmax=1.0)
        for C0 in C0_range:
            C_solved = odeint(dcdt_fixedA, C0, t,)
            ax.plot(t,C_solved,label=rf"$C_0=${C0:.1f}",color=cmap3(norm(C0)),)
        ax.set_ylabel('C(t)')
    axs[-1].set_xlabel('t')
    fig.tight_layout()
    return fig,axs

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
A_crit_value=calc_A_crit(x, r, m, b)

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

def cmap2(vmin, vmax, vcenter):
    if vcenter == vmin:
        return LinearSegmentedColormap.from_list("green_yellow", colors2[:-1]).reversed()
    elif vcenter == vmax:
        return LinearSegmentedColormap.from_list("yellow_red", colors2[1:]).reversed()
    else:
        return LinearSegmentedColormap.from_list("green_yellow_red", colors2).reversed()

def normation(vmin,vcenter,vmax):
    if vcenter == vmin or vcenter == vmax:
        return Normalize(vmin=vmin, vmax=vmax)
    else:
        return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


output_file = "results/noarrows.png"
output_file_arrows = "results/arrows.png"
output_file_potential = "results/potential.png"
output_file_comparison = "results/comparison.png"
output_file_trajectories = "results/trajectories.png"

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

fig,_ = make_plot_basin_vs_linear_stability(A=A,
                                            #C_F=calc_C_F(x=x,r=r),
                                            A_crit=calc_A_crit(x=x,r=r,m=m,b=b),
                                            basin_stability=calc_basin_stability(A=A, A_crit=calc_A_crit(x=x,r=r,m=m,b=b), C_crit=calc_C_crit(A=A, m=m, b=b)),
                                            linear_stability=calc_linear_stability(x=x, r=r, C_crit=calc_C_crit(A=A, m=m, b=b), C=calc_C_F(x=x,r=r)),
                                            #colors=colors,
                                            #title=title,
                                            #figsize=figsize,
                                            )
fig.savefig(output_file_comparison, dpi=300)
plt.close()

fig,_ = make_plot_trajectories(A_range=np.linspace(A_min,A_max,5),
                               C0_range=np.linspace(0,1,11),
                               func_C_crit=partial(calc_C_crit,m=m,b=b),
                               C_F=calc_C_F(x=x,r=r),
                               t=np.linspace(0, 7, 200),
                               figsize=(6,12),
                               )
fig.savefig(output_file_trajectories, dpi=300)
plt.close()

